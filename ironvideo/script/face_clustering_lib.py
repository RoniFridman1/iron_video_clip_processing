import sklearn

import utils
from typing import Any, Mapping, Optional, List
import os
import logging
import numpy as np
import shutil
from collections import defaultdict
from sklearn.cluster import DBSCAN
from deepface import DeepFace
import glob
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point
import segment_anything as sam
import cv2
import torch


PICTURES_SUFFIXES = ['jpg', 'png']
_DEFAULT_EMBEDDINGS_MODEL = 'Facenet'
_DEFAULT_DISTANCE_METRIC = 'euclidean_l2'
_DEFAULT_EUC_DISTANCE_THERSHOLD = 0.82


def get_embeddings(image_path:str) -> Optional[np.array]:
    if not any(image_path.endswith(ext) for ext in PICTURES_SUFFIXES):
        return None
    try:
        face_data = DeepFace.represent(
            img_path=image_path, model_name=_DEFAULT_EMBEDDINGS_MODEL, enforce_detection=False)
    except Exception as e:
        logging.error(f'Exception while geting embeddings for {image_path}: {e}')
        return None
    if not face_data:
        logging.error(f'Failed getting embeddings for {image_path}.')
        return None
    return np.array(face_data[0].get('embedding'))


def get_embeddings_per_file_name(pictures_folder: str) -> Mapping[str, np.array]:
    file_name_to_embeddings = dict()
    for image_file in os.listdir(pictures_folder):
        image_path = os.path.join(pictures_folder, image_file)
        embeddings = get_embeddings(image_path)
        if not embeddings is None:
            file_name_to_embeddings[image_file] =embeddings
    return file_name_to_embeddings


def get_frame_number(frame_image_name: str) -> int:
    return int(frame_image_name.split('_')[-1].split('.')[0])


def split_frame_seq_into_labels(all_pictures_folder_path: str) -> Mapping[int, List[str]]:
    frame_image_names = [file for file in os.listdir(all_pictures_folder_path) if file.endswith('jpg')]
    frame_image_names = sorted(frame_image_names,key=get_frame_number)
    label_to_frames = defaultdict(list)
    id = 0
    current_id = id
    label_to_frames[id].append(frame_image_names[0])
    for frame, next_frame in zip(frame_image_names[:-1], frame_image_names[1:]):
        data = safe_get_deepface_verify(
            os.path.join(all_pictures_folder_path, frame),
            os.path.join(all_pictures_folder_path, next_frame)
        )
        if data is None:
            continue
        if not data['verified']:
            in_previous_labels = False
            for tmp_id, frames in label_to_frames.items():
                data = safe_get_deepface_verify(
                    os.path.join(all_pictures_folder_path, next_frame),
                    os.path.join(all_pictures_folder_path, frames[-1])
                )
                if data is None:
                    continue
                if data['verified']:
                    current_id = tmp_id
                    label_to_frames[current_id].append(next_frame)
                    in_previous_labels = True
            if not in_previous_labels:
                current_id = tmp_id + 1
        label_to_frames[current_id].append(next_frame)
    return label_to_frames


def suggest_folder_split( file_name_to_embeddings, image_folder_path: str) -> Mapping[int, List[str]]:
    label_to_file_names = defaultdict(list)

    clt = DBSCAN(metric=_DEFAULT_DISTANCE_METRIC.split("_").pop(), n_jobs=-1)
    clt.fit(np.array([v for v in file_name_to_embeddings.values()]))
    for image_name, image_label in zip(file_name_to_embeddings.keys(), clt.labels_):
        if image_label <= -1: #Ignoring -1 because it's too far
            continue
        label_to_file_names[image_label].append(image_name)

    if len(label_to_file_names) < 1: # No labels here may mean that all distances were -1, which is "too far"
        label_to_file_names = split_frame_seq_into_labels(image_folder_path)
    return label_to_file_names


def maybe_split_person_folder(output_dir: str, person_folder_name: str, person_id_start: int) -> int:
    """Returns the next free person id to use."""
    image_folder_path = os.path.join(output_dir, person_folder_name, utils.ALL_PICS_FOLDER_NAME)
    file_name_to_embeddings = get_embeddings_per_file_name(image_folder_path)
    if not file_name_to_embeddings:
        logging.error(f'No embeddings extracted for: {image_folder_path}')
        return person_id_start
    label_to_file_names = suggest_folder_split(file_name_to_embeddings, image_folder_path)
    if len(label_to_file_names) < 2:
        logging.info(f'Not splitting {person_folder_name}.')
        return person_id_start
    logging.info(f'Splitting a {person_folder_name} acoording to labels.')
    for label, image_names in label_to_file_names.items():
        new_folder_name = f'{utils.DEFAULT_PERSON_FOLDER_NAME_PREFIX}_{person_id_start}'
        logging.info(f'Splitting {person_folder_name} label {label} to {new_folder_name}')
        print(f'Splitting {person_folder_name} label {label} to {new_folder_name}')
        for all_pictues_folder in utils.get_all_pictures_folder_names():
            new_dest = os.path.join(output_dir, new_folder_name, all_pictues_folder)
            img_folder_src = os.path.join(output_dir, person_folder_name, all_pictues_folder)
            person_id_start += 1
            os.makedirs(new_dest, exist_ok=True)
            for image_name in image_names:
                utils.copy_img_and_json(os.path.join(img_folder_src, image_name), new_dest)
    shutil.rmtree(os.path.join(output_dir, person_folder_name))
    return person_id_start


def safe_get_deepface_verify(img_path: str, other_img_path: str) -> Optional[Mapping[str, Any]]:
    try:
        return DeepFace.verify(
            img1_path=img_path, 
            img2_path=other_img_path,
            model_name=_DEFAULT_EMBEDDINGS_MODEL,
            enforce_detection=False,
            align=False,
            distance_metric=_DEFAULT_DISTANCE_METRIC,
            )
    except ValueError as e:
        logging.error('While checking if same person - {image_path}, {other_image_path}: {e}')
        return None


def is_same_person(img_path: str, other_img_path: str, distance_threshold: float = _DEFAULT_EUC_DISTANCE_THERSHOLD) -> bool:
    data = safe_get_deepface_verify(img_path, other_img_path)
    if data is None:
        return False
    return data['distance'] < min(distance_threshold, _DEFAULT_EUC_DISTANCE_THERSHOLD)

def calc_dist_representing_centers(centroid_1, centroid_2, metric):
    if metric == 'euclidean':
        return np.linalg.norm(centroid_1-centroid_2,ord=1)
    elif metric == 'cosine':
        return np.dot(centroid_1, centroid_2) / (np.linalg.norm(centroid_1) * np.linalg.norm(centroid_1))
    elif metric == 'l2':
        return np.linalg.norm(centroid_1 - centroid_2,ord=2)
    elif metric == 'convex_hulls':
        # centroid_1 is the polygon, centroid_2 is the points of the other person
        shapely_points = [Point(point) for point in centroid_2]
        # Check containment and count the points inside the polygon
        dist = sum(1 for point in shapely_points if centroid_1.contains(point))
        return dist
    elif metric == 'tensor_l2':
        return torch.cdist(centroid_1,centroid_2,p=2)
    elif metric == 'tensor_pairwise':
        pdist = torch.nn.PairwiseDistance(p=2)
        return pdist(centroid_1,centroid_2)
    else:
        return None

def calc_representing_embedding(embeddings_list, method='centroid'):
    if method == 'centorid':
        return np.mean(embeddings_list, axis=0)
    elif method == 'medoid':
        return np.median(embeddings_list, axis=0)
    elif method == 'convex_hull':
        cv = ConvexHull(np.array(embeddings_list))
        return Polygon(cv.points[cv.vertices])
    elif method == 'tensor_mean':
        return torch.mean(embeddings_list)
    elif method == 'tensor_median':
        return torch.median(embeddings_list)
    else:
        return None


def merge_persons_folders(output_dir:str, person_folder: str, other_person_folder: str):
    person_folder_path = os.path.join(output_dir, person_folder)
    other_person_folder_path = os.path.join(output_dir, other_person_folder)
    image_names = os.listdir(os.path.join(other_person_folder_path, utils.ALL_PICS_FOLDER_NAME))
    for image_name in image_names:
        for all_pictures_folder in utils.get_all_pictures_folder_names():
            utils.copy_img_and_json(os.path.join(other_person_folder_path, all_pictures_folder, image_name),
                                    os.path.join(person_folder_path, all_pictures_folder))
    shutil.rmtree(os.path.join(other_person_folder_path, utils.MAIN_PIC_FOLDER_NAME))
    shutil.rmtree(os.path.join(other_person_folder_path, utils.SEARCH_PICS_FOLDER_NAME))
    shutil.rmtree(os.path.join(other_person_folder_path))


def merge_similar_persons(output_dir: str, persons_metadata: Mapping[str, Any]|None) -> Mapping[str, Any]|None:
    """
    :param output_dir: Main folder of video
    :param persons_metadata:
    :return: persons_metadata.
    """
    if persons_metadata is None:
        logging.error(f'Person metadata is None for {output_dir}. Will not compute merging.')
        return persons_metadata
    logging.info('Finding similarity threshold')
    print("Start merging")
    logging.info('Checking all persons folders for matchs.')
    person_folders = sorted(os.listdir(output_dir),key=utils.get_person_id)
    representing_emb_of_person = {}
    already_checked = []
    distance_between_centroids = {}
    persons_embeddings = {}
    sam_predictor = sam.SamPredictor(sam.sam_model_registry['vit_h'](checkpoint="./sam_vit_h_4b8939.pth"))
    for person_folder in person_folders:
        if not person_folder.startswith(utils.DEFAULT_PERSON_FOLDER_NAME_PREFIX):
            continue
        person_embeddings = []
        # for image in glob.glob(os.path.join(output_dir, person_folder, utils.ALL_PICS_FOLDER_NAME)+"/*.jpg"):
        #     sam_predictor.set_image(cv2.imread(image, cv2.IMREAD_COLOR), image_format='RGB')
        #     person_embeddings.append(sam_predictor.get_image_embedding())
        person_embeddings = [
                                np.array(
                                DeepFace.represent(
                                img_path=image,
                                model_name=_DEFAULT_EMBEDDINGS_MODEL,
                                enforce_detection=False,
                                detector_backend='opencv',  # options: opencv, retinaface, mtcnn, ssd, dlib or mediapipe
                                normalization='Facenet',  # options: base, raw, Facenet
                                align=False)[0]['embedding']) for image in glob.glob(os.path.join(output_dir,
                                                        person_folder, utils.ALL_PICS_FOLDER_NAME)+"/*.jpg")
                            ]
        persons_embeddings[person_folder] = person_embeddings
        representing_emb_of_person[person_folder] = calc_representing_embedding(person_embeddings, method='centorid')
    utils.plot_points_from_dict(persons_embeddings)

    for i,first_person in enumerate(person_folders):
        for second_person in person_folders[i+1:]:
            if first_person == second_person or {first_person, second_person} in already_checked:
                continue
            dist = calc_dist_representing_centers(representing_emb_of_person[first_person], representing_emb_of_person[second_person],
                                                  metric='l2')
            already_checked.append({first_person,second_person})
            distance_between_centroids[dist] = (first_person, second_person)

    # Merge folders if they are below the distance between their centroids is below 70% of average

    distance_threshold = np.mean([*distance_between_centroids.keys()])
    for dist, pair in sorted(distance_between_centroids.items()):
        if not os.path.exists(os.path.join(output_dir,pair[0])) or not os.path.exists(os.path.join(output_dir,pair[1]))\
                or dist > distance_threshold:
            continue
        merge_persons_folders(output_dir, pair[0], pair[1])
        for merged_folder in [pair[0], pair[1]]:
            if merged_folder in persons_metadata:
                del persons_metadata[utils.get_person_id(merged_folder)]
    person_folders_after_merge = sorted(os.listdir(output_dir), key=utils.get_person_id)
    print(f"Distance Treshold: {distance_threshold}")
    print(f"Before merging: count={len(person_folders)} names={person_folders}"
          f"\nAfter merging: count={len(person_folders_after_merge)} names={person_folders_after_merge}")
    print(sorted(distance_between_centroids.items()))
    return persons_metadata


def get_avg_max_distances_per_person(output_dir: str) -> float:
    distances = []
    for person_dir in os.listdir(output_dir):
        if not person_dir.startswith(utils.DEFAULT_PERSON_FOLDER_NAME_PREFIX):
            continue
        all_pictures_path = os.path.join(output_dir,person_dir,utils.ALL_PICS_FOLDER_NAME)
        image_names = os.listdir(all_pictures_path)
        try:
            data = DeepFace.verify(
            img1_path=os.path.join(all_pictures_path,image_names[0]), 
            img2_path=os.path.join(all_pictures_path,image_names[-1]), 
            model_name=_DEFAULT_EMBEDDINGS_MODEL,
            enforce_detection=False,
            align=False,
            distance_metric=_DEFAULT_DISTANCE_METRIC)
        except Exception:
            continue
        distances.append(data['distance'])
    return(sum(distances)/max(len(distances), 1))
