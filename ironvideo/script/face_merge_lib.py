from typing import Any, Mapping
import os
import logging
import numpy as np
import shutil

from deepface import DeepFace
import glob
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point
import torch

# from . import utils
import utils
_DEFAULT_EMBEDDINGS_MODEL = "Facenet"


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


def merge_persons_folders(output_dir: str, person_folder: str, other_person_folder: str):
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


def calc_dist_representing_centers(centroid_1, centroid_2, metric):
    if metric == 'euclidean':
        return np.linalg.norm(centroid_1 - centroid_2, ord=1)
    elif metric == 'cosine':
        return np.dot(centroid_1, centroid_2) / (np.linalg.norm(centroid_1) * np.linalg.norm(centroid_1))
    elif metric == 'l2':
        return np.linalg.norm(centroid_1 - centroid_2, ord=2)
    elif metric == 'convex_hulls':
        # centroid_1 is the polygon, centroid_2 is the points of the other person
        shapely_points = [Point(point) for point in centroid_2]
        # Check containment and count the points inside the polygon
        dist = sum(1 for point in shapely_points if centroid_1.contains(point))
        return dist
    elif metric == 'tensor_l2':
        return torch.cdist(centroid_1, centroid_2, p=2)
    elif metric == 'tensor_pairwise':
        pdist = torch.nn.PairwiseDistance(p=2)
        return pdist(centroid_1, centroid_2)
    else:
        return None


def merge_similar_persons(output_dir: str, persons_metadata: Mapping[str, Any] | None) -> tuple[Mapping[str, Any], dict[
    str, int | dict[Any, Any] | list[Any] | Any]] | None:
    """
    :param output_dir: Main folder of video
    :param persons_metadata:
    :return: persons_metadata.
    """
    if persons_metadata is None:
        logging.error(f'Person metadata is None for {output_dir}. Will not compute merging.')
        return persons_metadata
    print("Start merging")
    merge_metadata = {}
    person_folders = sorted(os.listdir(output_dir), key=utils.get_person_id)
    before_merging_length = len(person_folders)
    merged_pairs = {}

    representing_emb_of_person = {}
    already_checked = []
    distance_between_centroids = {}
    persons_embeddings = {}
    for person_folder in person_folders:
        if not person_folder.startswith(utils.DEFAULT_PERSON_FOLDER_NAME_PREFIX):
            continue
        main_folder_for_embedding = os.path.join(output_dir, person_folder, utils.SEARCH_PICS_FOLDER_NAME)
        if len(os.listdir(main_folder_for_embedding)) / 2 < 3:
            main_folder_for_embedding = os.path.join(output_dir, person_folder, utils.ALL_PICS_FOLDER_NAME)
        if len(os.listdir(main_folder_for_embedding)) / 2 < 3:
            continue

        person_embeddings = [
            np.array(
                DeepFace.represent(
                    img_path=image,
                    model_name=_DEFAULT_EMBEDDINGS_MODEL,
                    enforce_detection=False,
                    detector_backend='opencv',  # options: opencv, retinaface, mtcnn, ssd, dlib or mediapipe
                    normalization='Facenet',  # options: base, raw, Facenet
                    align=True)[0]['embedding']) for image in glob.glob(main_folder_for_embedding + "/*.jpg")
        ]
        persons_embeddings[person_folder] = person_embeddings
    person_folders = sorted([*persons_embeddings.keys()])
    utils.plot_points_from_dict(persons_embeddings)
    persons_embeddings = utils.reduce_embeddings_dim_via_pca(persons_embeddings, 5)
    for person_folder in person_folders:
        representing_emb_of_person[person_folder] = calc_representing_embedding(persons_embeddings[person_folder],
                                                                                method='medoid')

    for i, first_person in enumerate(person_folders):
        for second_person in person_folders[i + 1:]:
            if first_person == second_person or {first_person, second_person} in already_checked:
                continue
            dist = calc_dist_representing_centers(representing_emb_of_person[first_person],
                                                  representing_emb_of_person[second_person],
                                                  metric='l2')
            already_checked.append({first_person, second_person})
            distance_between_centroids[dist] = (first_person, second_person)

    # Merge folders if they are below the distance between their centroids is below the distance threshold
    distance_threshold = np.mean([*distance_between_centroids.keys()]) / 3.0
    for dist, pair in sorted(distance_between_centroids.items()):
        if not os.path.exists(os.path.join(output_dir, pair[0])) or not os.path.exists(
                os.path.join(output_dir, pair[1])) \
                or dist > distance_threshold:
            continue
        merged_pairs[(pair)] = dist
        merge_persons_folders(output_dir, pair[0], pair[1])
        for merged_folder in [pair[0], pair[1]]:
            if merged_folder in persons_metadata:
                del persons_metadata[utils.get_person_id(merged_folder)]
    person_folders_after_merge = sorted(os.listdir(output_dir), key=utils.get_person_id)
    print(f"Distance Treshold: {distance_threshold}")
    merge_metadata['n_people_after_merge'] = len(person_folders_after_merge)
    merge_metadata['distance_threshold_used'] = distance_threshold
    merge_metadata['merged folders'] = merged_pairs

    for k,v in representing_emb_of_person.items():
        representing_emb_of_person[k] = str(v)
    merge_metadata['representing_embedding_of_person'] = representing_emb_of_person
    merge_metadata['all_distances_between_rep_embeddings'] = sorted(distance_between_centroids.items())
    return persons_metadata, merge_metadata


