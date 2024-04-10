from typing import Any, Mapping, Optional, List
import os
import logging
import numpy as np
import shutil
from collections import defaultdict
from sklearn.cluster import DBSCAN
from deepface import DeepFace

# from . import utils
import utils
PICTURES_SUFFIXES = ['jpg', 'png']
_DEFAULT_EMBEDDINGS_MODEL = 'Facenet'
_DEFAULT_DISTANCE_METRIC = 'euclidean_l2'
_DEFAULT_EUC_DISTANCE_THERSHOLD = 0.82


def get_embeddings(image_path: str) -> Optional[np.array]:
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
            file_name_to_embeddings[image_file] = embeddings
    return file_name_to_embeddings


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


def suggest_folder_split(file_name_to_embeddings, image_folder_path: str) -> Mapping[int, List[str]]:
    label_to_file_names = defaultdict(list)

    clt = DBSCAN(metric=_DEFAULT_DISTANCE_METRIC.split("_").pop(), n_jobs=-1)
    clt.fit(np.array([v for v in file_name_to_embeddings.values()]))
    for image_name, image_label in zip(file_name_to_embeddings.keys(), clt.labels_):
        if image_label <= -1:  # Ignoring -1 because it's too far
            continue
        label_to_file_names[image_label].append(image_name)

    if len(label_to_file_names) < 1:  # No labels here may mean that all distances were -1, which is "too far"
        label_to_file_names = split_frame_seq_into_labels(image_folder_path)
    return label_to_file_names


def split_frame_seq_into_labels(all_pictures_folder_path: str) -> Mapping[int, List[str]]:
    frame_image_names = [file for file in os.listdir(all_pictures_folder_path) if file.endswith('jpg')]
    frame_image_names = sorted(frame_image_names, key=utils.get_frame_number)
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
