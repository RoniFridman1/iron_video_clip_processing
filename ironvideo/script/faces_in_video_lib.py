from argparse import Namespace
from pathlib import Path
from typing import Any, Optional, Mapping
import json
import os
import cv2
import logging

from insightface.app import FaceAnalysis

from yolo_tracking.examples.track import run as track_run
# from . import utils
# from . import best_faces_lib
# from . import face_merge_lib
# from . import face_split_lib

import utils
import best_faces_lib
import face_merge_lib
import face_split_lib


# import face_clustering_lib

# YOLO Model default parameters
DEFAULT_YOLO_MODEL = 'yolov8n.pt'
DEFAULT_TRACKING_METHOD = 'botsort'
DEFAULT_REID_MODEL = 'osnet_x1_0_dukemtmcreid.pt'

DEFAULT_PREFIX_SOURCE_URL = "https://vid.israeltechguard.org/source-vid/"
DEFAULT_PREFIX_OUTPUT_URL = "https://vid.israeltechguard.org/output-vid/"


def protect(x, op):
    try:
        return op(x)
    except:
        return 'yo'


def crop_frame(frame, frame_coordinates):
    return frame[int(frame_coordinates[1]):int(frame_coordinates[3]),
           int(frame_coordinates[0]):int(frame_coordinates[2])]


def crop_face(cropped_frame, safe_face_coords):
    return cropped_frame[safe_face_coords['bottom']: safe_face_coords['top'],
           safe_face_coords['left']: safe_face_coords['right']]


def save_person_frame(person_id, frame_id, output_dir, cropped_face, safe_face_coords,
                      margin_ratio_cropped_pairs):
    person_folder_name = utils.get_person_folder_name(person_id)

    output_path = os.path.join(output_dir, person_folder_name, utils.ALL_PICS_FOLDER_NAME)
    utils.create_folders_per_person_save_image_and_json(output_path, frame_id, cropped_face, safe_face_coords)

    for margin_ratio in utils.DEFAULT_CROP_MARGIN_RATIO_LIST:
        cropped_face_margin_ratio, safe_face_coords_margin_ratio = margin_ratio_cropped_pairs[margin_ratio]
        margin_ration_all_images_path = os.path.join(output_dir, person_folder_name,
                                                     utils.ALL_PICS_RATIO_MARGIN_PREFIX_FOLDER_NAME + f"_{int(margin_ratio * 100)}")
        utils.create_folders_per_person_save_image_and_json(margin_ration_all_images_path, frame_id,
                                                            cropped_face_margin_ratio,
                                                            safe_face_coords_margin_ratio)


def get_face_analysis():
    face_analysis = FaceAnalysis(allowed_modules=['detection'])
    face_analysis.prepare(ctx_id=0)
    return face_analysis


def find_and_save_humans_in_frame(frame_id, frame, frame_coords, frame_classes, frame_ids, output_dir,
                                  face_analysis=None):
    # print(f"frame ID:\t{frame_id}")
    if len(frame_coords) == 0:
        logging.warning(f"No coordinates provided for frame {frame_id}. Skipping.")
        return
    for obj_id, obj_coords in enumerate(frame_coords):
        if frame_classes[obj_id] != 0:
            continue
        cropped_frame = crop_frame(frame, obj_coords)
        if face_analysis is None:
            face_analysis = get_face_analysis()
        faces = face_analysis.get(cropped_frame)
        if len(faces) == 0:
            continue

        # If more than one face was found, will choose the one with the biggest area.
        chosen_face = max(faces, key=lambda x: (x.bbox[3] - x.bbox[1]) * (x.bbox[2] - x.bbox[0]))

        face_coords_on_frame = utils.create_frame_face_coors(obj_coords, chosen_face.bbox)
        safe_face_coords = utils.get_safe_face_coords(frame, face_coords_on_frame)
        cropped_face = crop_face(frame, safe_face_coords)

        # Margin Ratio
        margin_ratio_cropped_pairs = {}
        for ratio in utils.DEFAULT_CROP_MARGIN_RATIO_LIST:
            safe_face_coords_margin_ratio = utils.get_safe_face_coords(frame, face_coords_on_frame, ratio=ratio)
            cropped_face_margin_ratio = crop_face(frame, safe_face_coords_margin_ratio)
            margin_ratio_cropped_pairs[ratio] = (cropped_face_margin_ratio, safe_face_coords_margin_ratio)

        person_id = None

        ## TODO: refactor
        if isinstance(frame_ids, str) and frame_ids == 'yo':
            continue
        try:
            person_id = frame_ids[obj_id]
            save_person_frame(
                person_id=person_id,
                frame_id=frame_id,
                output_dir=output_dir,
                cropped_face=cropped_face,
                safe_face_coords=face_coords_on_frame,
                margin_ratio_cropped_pairs=margin_ratio_cropped_pairs,
            )
        except (SystemExit, KeyboardInterrupt):
            raise
        except Exception as e:
            logging.exception(
                f'Ran into an exception while saving pictures of person_{person_id} in frame {frame_id}!')


def get_humans_from_frames(video_path,
                           output_dir,
                           coordinates_arr,
                           object_id_arr,
                           classes_arr,
                           compute_time_dict,
                           output_version='v0001',
                           max_frames=float('inf'),
                           face_analysis=None,
                           enable_postprocess_split=False) -> tuple[dict[str | Any], dict[str | Any],dict[str | Any]]:
    """Returns video metadata."""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Couldn't open the video.")

    frames_per_second = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_idx = 0
    with utils.TimeTracker("insight_ai_find_and_save_humans_in_frame", compute_time_dict):
        while frame_idx < frame_count:
            # if frame_idx % 2 != 0:
            #     frame_idx += 1
            #     continue
            logging.info('frame_idx: %s', frame_idx)
            print(f"frame_idx: {frame_idx}")
            ret, frame = cap.read()
            if not ret:
                break
            find_and_save_humans_in_frame(
                frame_id=frame_idx,
                frame=frame,
                frame_coords=coordinates_arr[frame_idx],
                frame_classes=classes_arr[frame_idx],
                frame_ids=object_id_arr[frame_idx],
                output_dir=output_dir,
                face_analysis=face_analysis)
            frame_idx += 1
        cap.release()
    splitting_metadata = {}
    splitting_metadata['n_people_after_tracker'] = utils.count_person_folders(output_dir)

    with utils.TimeTracker("splitting person folders", compute_time_dict):
        if enable_postprocess_split:
            logging.info('Maybe splitting folders accoreding to embeddings.')

            next_usable_id = int(sorted([person_folder.split("_").pop() for person_folder in os.listdir(output_dir)
                              if person_folder.startswith(utils.DEFAULT_PERSON_FOLDER_NAME_PREFIX)]).pop()) + 1
            next_usable_id = max(1000, next_usable_id)

            for person_folder_name in os.listdir(output_dir):
                logging.info(f'Maybe splitting {person_folder_name}...')
                print(f'Maybe splitting {person_folder_name}...')
                if not person_folder_name.startswith(utils.DEFAULT_PERSON_FOLDER_NAME_PREFIX):
                    continue
                next_usable_id = face_split_lib.maybe_split_person_folder(output_dir, person_folder_name,
                                                                          next_usable_id)
    splitting_metadata['n_people_after_split'] = utils.count_person_folders(output_dir)


    video_name, ext = os.path.splitext(os.path.basename(video_path))
    output_url = f'{DEFAULT_PREFIX_OUTPUT_URL}{video_name}/{output_version}'
    video_url = f'{DEFAULT_PREFIX_SOURCE_URL}{video_name}{ext}'
    video_metadata = {"video_url": video_url, "output_url": output_url, "video_fps": frames_per_second,
            "video_width": video_width, "video_height": video_height, "video_frame_count": frame_count}

    return video_metadata, compute_time_dict, splitting_metadata


def classify_video_with_yolo(video_path, frames_per_sec=None):
    # 'yolo_model' has to include yolov8 or yolox.
    # i.e. yolov8n (smallest) or switch "n" with s,m,l,x (bigger models and slower computation time but more accurate).
    yolo_model = DEFAULT_YOLO_MODEL
    tracking_method = DEFAULT_TRACKING_METHOD  # choose from 'botsort','bytetrack'

    # ReID Model: Can choose alot of other models, and datasets it was trained on. i.e., osnet_x1_0_<dataset_name>
    # Datasets include: market1501, msmt17, vehicleid,dukemtmcreid
    reid_model = DEFAULT_REID_MODEL

    args = Namespace(yolo_model=Path(yolo_model), source=video_path, show=True, stream=True,
                     tracking_method=tracking_method,
                     reid_model=Path(reid_model), half=False, per_class=False,
                     tracker=f'yolo_tracking/boxmot/configs/{tracking_method}.yaml')
    tags = track_run(args)

    coordinates_arr = []
    ids = []
    obj_class = []
    obj_mask = []
    for tag in tags:
        coordinates_arr.append(protect(tag, lambda x: x.boxes.xyxy.cpu().numpy()))
        ids.append(protect(tag, lambda x: x.boxes.id.cpu().numpy()))
        obj_class.append(protect(tag, lambda x: x.boxes.cls.cpu().numpy()))
        obj_mask.append(protect(tag, lambda x: x.masks.xy))

    return coordinates_arr, ids, obj_class, obj_mask


def copy_best_images_and_json(source_person_folder_name: str, output_path: Optional[str] = None) -> int:
    if output_path is None:
        output_path = source_person_folder_name
    os.makedirs(output_path, exist_ok=True)

    search_picture_folder_path = os.path.join(output_path, utils.SEARCH_PICS_FOLDER_NAME)
    os.makedirs(search_picture_folder_path, exist_ok=True)

    main_picture_folder_path = os.path.join(output_path, utils.MAIN_PIC_FOLDER_NAME)
    os.makedirs(main_picture_folder_path, exist_ok=True)

    best_image_paths = best_faces_lib.get_best_image_paths(source_person_folder_name)

    if not best_image_paths:
        logging.warning(f"No good photos were found for {source_person_folder_name}! Using top resolution pics.")
        best_image_paths = best_faces_lib.get_top_resolution_images_per_person(source_person_folder_name, 10)

    main_image_path = best_image_paths[0]
    utils.copy_img_and_json(main_image_path, main_picture_folder_path, utils.MAIN_PIC_FILE_NAME)

    for ratio in utils.DEFAULT_CROP_MARGIN_RATIO_LIST:
        main_image_path_ratio = main_image_path.replace(utils.ALL_PICS_FOLDER_NAME,
                                                        utils.ALL_PICS_RATIO_MARGIN_PREFIX_FOLDER_NAME + f"_{int(100 * ratio)}")
        main_pic_file_name_ratio = utils.MAIN_PIC_FILE_NAME.split('.')[0] + f"_{int(100 * ratio)}.jpg"
        utils.copy_img_and_json(main_image_path_ratio, main_picture_folder_path, main_pic_file_name_ratio)

    for path in best_image_paths:
        utils.copy_img_and_json(path, search_picture_folder_path)
    main_image_frame_id = int(os.path.basename(main_image_path).split('.')[0].split('_')[-1])
    return main_image_frame_id


def extract_best_faces_per_person(
        source_output_dir: str,
        dest_output_dir: Optional[str] = None,
        persons_metdata: Optional[Mapping[str, Any]] = None) -> Optional[Mapping[str, Any]]:
    "Returns all persons metadata."
    if not os.path.exists(source_output_dir):
        logging.warning(
            'Can\'t extract best image cause source dir does not exists! This can happen if we had a problem openning the video, or, not faces were found in the video!')
        return None
    if dest_output_dir is None:
        dest_output_dir = source_output_dir
    persons_metdata = persons_metdata or dict()
    for person_dir in os.listdir(source_output_dir):
        if not person_dir.startswith(utils.DEFAULT_PERSON_FOLDER_NAME_PREFIX):
            continue
        if os.path.exists(os.path.join(source_output_dir, person_dir, utils.MAIN_PIC_FOLDER_NAME)):
            print(f"{person_dir} already has a main picture..")
            continue
        person_data = dict()
        person_path = os.path.join(source_output_dir, person_dir)
        person_data['rects'] = save_bboxes_of_person(person_path)
        try:
            main_image_frame_id = copy_best_images_and_json(
                source_person_folder_name=person_path,
                output_path=os.path.join(dest_output_dir, person_dir))
            main_pic_folder_path = os.path.join(person_path, utils.MAIN_PIC_FOLDER_NAME)
            person_data['main_image'] = os.path.join(main_pic_folder_path, utils.MAIN_PIC_FILE_NAME)

            for ratio in utils.DEFAULT_CROP_MARGIN_RATIO_LIST:
                main_pic_file_name_ratio = utils.MAIN_PIC_FILE_NAME.split('.')[0] + f"_{int(100 * ratio)}.jpg"
                person_data[f'main_image_{int(100 * ratio)}'] = os.path.join(main_pic_folder_path,
                                                                             main_pic_file_name_ratio)
                utils.create_gif(
                    input_folder=os.path.join(person_path,
                                              utils.ALL_PICS_RATIO_MARGIN_PREFIX_FOLDER_NAME + f"_{int(ratio * 100)}"),
                    output_path=os.path.join(person_path, utils.MAIN_PIC_FOLDER_NAME,
                                             f"main_gif_{int(ratio * 100)}.gif"),
                    selected_frame_number=main_image_frame_id)
        except (SystemExit, KeyboardInterrupt):
            raise
        except Exception:
            logging.exception(f"Failed for {person_dir}, skipping.")
        persons_metdata[utils.get_person_id(person_dir)] = person_data
    return persons_metdata


# TODO: refactor
def save_bboxes_of_person(person_id_sub_folder):
    person_all_pictures_folder = os.path.join(person_id_sub_folder, utils.ALL_PICS_FOLDER_NAME)
    bboxes = {}
    for file in next(os.walk(person_all_pictures_folder))[2]:
        filename, ext = os.path.splitext(file)
        if ext != '.json':
            continue
        frame_id = int(filename.split("_")[-1])
        bboxes[frame_id] = json.load(open(os.path.join(person_all_pictures_folder, file), ))
    return bboxes


class ProcessVideo():
    def __init__(self, *, enable_codeformer: bool) -> None:
        self._face_analysis = get_face_analysis()
        self._enable_codeformer = enable_codeformer
        if self._enable_codeformer:
            self._codeformer = best_faces_lib.CodeFormerRestoreFace()  # Init CodeFormer

    def naive_process_video(
            self,
            video_path: str,
            output_dir: str,
            output_version: str = 'v0001',
            enable_postprocess_split: bool = False,
            enable_postprocess_merge: bool = False) -> Mapping[str, Any]:
        compute_time = {}
        with utils.TimeTracker('yolo', compute_time):
            coordinates_arr, ids, obj_class, _ = classify_video_with_yolo(video_path)

        video_metadata, compute_time, split_metadata = get_humans_from_frames(
            video_path=video_path,
            output_dir=output_dir,
            coordinates_arr=coordinates_arr,
            object_id_arr=ids,
            classes_arr=obj_class,
            output_version=output_version,
            face_analysis=self._face_analysis,
            enable_postprocess_split=enable_postprocess_split,
            compute_time_dict=compute_time)


        with utils.TimeTracker('insight_ai_extract_best_faces', compute_time):
            person_metadata = extract_best_faces_per_person(output_dir)

        if enable_postprocess_merge and person_metadata:
            with utils.TimeTracker('merging person folders', compute_time):
                person_metadata, merge_metadata = face_merge_lib.merge_similar_persons(output_dir, person_metadata)

            with utils.TimeTracker('insight_ai_extract_best_faces - after merging', compute_time):
                person_metadata = extract_best_faces_per_person(output_dir, persons_metdata=person_metadata)
        else:
            merge_metadata = {'n_people_after_merge':utils.count_person_folders(output_dir)}

        video_metadata['persons'] = person_metadata
        os.makedirs(output_dir, exist_ok=True)
        json.dump(video_metadata, open(os.path.join(output_dir, f"main_json.json"), 'w'))
        with open(os.path.join(output_dir, f"main_json.js"), 'w', encoding='utf8') as f:
            f.write("var main_json = " + str(video_metadata))
            f.close()
        if self._enable_codeformer:
            try:
                with utils.TimeTracker('code_former_restoration_duration', compute_time):
                    self._codeformer.restore_main_pictures(output_dir)
            except (SystemExit, KeyboardInterrupt):
                raise
            except Exception as e:
                logging.exception(f'Failed face restoration for {output_dir}')


        dashboard_metadata = {
            'n_frames': video_metadata['video_frame_count'],
            'video_width': video_metadata['video_width'],
            'video_height': video_metadata['video_height'],
            'n_people tracker,split,merge': (split_metadata['n_people_after_tracker'],
                                                  split_metadata['n_people_after_split'],
                                                  merge_metadata['n_people_after_merge']),
            'compute_time': compute_time,
            'split_metadata': split_metadata,
            'merge_metadata': merge_metadata
        }
        return dashboard_metadata
