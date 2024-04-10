from typing import Dict, List, Optional
import contextlib
import datetime
import logging
import numpy as np
import numpy.typing as npt
import cv2
import imageio
import glob
import json
import os
import shutil
import time
import pathlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

FRAMES_WITH_BBOXES_FOLDER_NAME = 'frames_with_bboxes'
ALL_PICS_FOLDER_NAME = 'all_pictures'
ALL_PICS_RATIO_MARGIN_PREFIX_FOLDER_NAME = f'all_pictures_ratio_margin'
SEARCH_PICS_FOLDER_NAME = 'search_picture'
MAIN_PIC_FOLDER_NAME = 'main_picture'
MAIN_PIC_FILE_NAME = 'main_pic.jpg'
DEFAULT_PERSON_FOLDER_NAME_PREFIX = 'person'

_DEFAULT_CROP_MARGIN = 0
DEFAULT_CROP_MARGIN_RATIO_LIST = [0.5, 1, 2]


def safe_crop_start(coord, margin=_DEFAULT_CROP_MARGIN):
    return max(0, int(coord) - margin)


def safe_crop_end(coord, max_len, margin=_DEFAULT_CROP_MARGIN):
    return min(int(coord) + margin, max_len)

def count_person_folders(output_dir):
    return len([person_folder for person_folder in os.listdir(output_dir)
                if person_folder.startswith(DEFAULT_PERSON_FOLDER_NAME_PREFIX)])
def get_safe_face_coords(cropped_frame, face_bbox, ratio=None):
    if ratio is not None:
        height_margin = int((face_bbox['top'] - face_bbox['bottom']) * ratio)
        width_margin = int((face_bbox['right'] - face_bbox['left']) * ratio)
        return {'bottom': safe_crop_start(face_bbox['bottom'], margin=height_margin),
                'top': safe_crop_end(face_bbox['top'], cropped_frame.shape[0], margin=height_margin),
                'left': safe_crop_start(face_bbox['left'], margin=width_margin),
                'right': safe_crop_end(face_bbox['right'], cropped_frame.shape[1], margin=width_margin)}
    else:
        return {'bottom': safe_crop_start(face_bbox['bottom']),
                'top': safe_crop_end(face_bbox['top'], cropped_frame.shape[0]),
                'left': safe_crop_start(face_bbox['left']),
                'right': safe_crop_end(face_bbox['right'], cropped_frame.shape[1])}


def get_person_id(person_folder_name: str) -> int:
    return int(person_folder_name.split('_')[-1])


def get_person_folder_name(person_index, prefix=DEFAULT_PERSON_FOLDER_NAME_PREFIX):
    return f'{prefix}_{int(person_index)}'


def get_json_from_img_path(img_path: str) -> str:
    return pathlib.Path(img_path).with_suffix('.json')


def copy_img_and_json(img_path: str, dest_path: str, rename_file_name: Optional[str] = None) -> None:
    shutil.copy2(img_path, dest_path)
    json_path = get_json_from_img_path(img_path)
    shutil.copy2(json_path, dest_path)

    # TODO: refactor
    if rename_file_name:
        dest_img_path = os.path.join(dest_path, os.path.basename(img_path))
        os.rename(dest_img_path, os.path.join(dest_path, rename_file_name))

        rename_json_name = os.path.splitext(rename_file_name)[0] + '.json'
        dest_json_path = os.path.join(dest_path, os.path.basename(json_path))
        os.rename(dest_json_path, os.path.join(dest_path, rename_json_name))


def get_all_picture_ratio_folder_name(ratio: float) -> str:
    return ALL_PICS_RATIO_MARGIN_PREFIX_FOLDER_NAME + f"_{int(ratio * 100)}"


def get_all_pictures_folder_names() -> List[str]:
    margin_ratio_folders = [
        get_all_picture_ratio_folder_name(margin_ratio)
        for margin_ratio in DEFAULT_CROP_MARGIN_RATIO_LIST]
    return margin_ratio_folders + [ALL_PICS_FOLDER_NAME]


def draw_bbox_on_image(
        img: npt.NDArray[np.uint8], bbox: Dict["str", int]
) -> npt.NDArray[np.uint8]:
    """
    Draw a bounding box on an image using OpenCV

    :param: img: the img array
    :param: bbox: Dict which contains bottom left and top right coordinates: (left,bottom): bottom left. (right,top): top right
    :return: img with bbox
    """

    x1, y1, x2, y2 = bbox["left"], bbox["bottom"], bbox["right"], bbox["top"]
    # Draw the bounding box on the image
    color = (0, 0, 255)  # BGR color (red)
    thickness = 2
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

    return img


def create_frame_face_coors(
        person_bbox_on_frame: npt.NDArray[np.float32], face_bbox_on_crop: Dict[int, int]
) -> Dict["str", int]:
    """
    :param person_bbox_on_frame: (x1,y1,x2,y2): (x1,y1) bottom left, (x2,y2) top right
    :param face_bbox_on_crop: Dict which contains bottom left and top right coordinates: (left,bottom): bottom left. (right,top): top right
    :return: Dict which contains bottom left and top right coordinates of the face in the original frame: (left,bottom): bottom left. (right,top): top right
    """
    return {
        "left": int(person_bbox_on_frame[0] + face_bbox_on_crop[0]),
        "bottom": int(person_bbox_on_frame[1] + face_bbox_on_crop[1]),
        "right": int(person_bbox_on_frame[0] + face_bbox_on_crop[2]),
        "top": int(person_bbox_on_frame[1] + face_bbox_on_crop[3]),
    }


def create_folders_per_person_save_image_and_json(output_path, frame_id, cropped_face, safe_face_coords=None):
    os.makedirs(output_path, exist_ok=True)
    cv2.imwrite(os.path.join(output_path, f"frame_{frame_id}.jpg"), cropped_face)
    if safe_face_coords is not None:
        json.dump(safe_face_coords, open(os.path.join(output_path, f"frame_{frame_id}.json"), 'w'))


def create_gif(input_folder, output_path, selected_frame_number=None):
    # Get a list of image files in the folder
    image_files = sorted(glob.glob(os.path.join(input_folder, "*.jpg")))
    selected_frame_idx = None
    for i, image_path in enumerate(image_files):
        if f"frame_{selected_frame_number}" in image_path:
            logging.info('%s: %s', image_path, i)
            selected_frame_idx = i

    if selected_frame_idx is None:
        logging.error("Failed to create GIF since the main image was found.")
        return

    # Calculate the range of frames to include in the GIF
    start_index = max(0, selected_frame_idx - 25 + 1)
    end_index = min(selected_frame_idx + 25, len(image_files) - 1)

    # Load the selected frames into memory
    selected_frames = []
    for i in range(start_index, end_index + 1):
        frame = cv2.imread(image_files[i])
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        selected_frames.append(frame)

    # Create the GIF from the selected frames
    with imageio.get_writer(output_path, mode='I', duration=1, loop=0) as writer:
        for frame in selected_frames:
            writer.append_data(frame)
    writer.close()


def format_time_duration(time_sec: float) -> str:
    return str(datetime.timedelta(seconds=time_sec))


@contextlib.contextmanager
def TimeTracker(name: str, dest: Dict):
    logging.info('Starting stage %s', name)
    start_time = time.time()
    try:
        yield
        duration = format_time_duration(time.time() - start_time)
        logging.info('Stage %s completed in %s', name, duration)
    except:
        duration = format_time_duration(time.time() - start_time)
        logging.exception('Stage %s failed after %s', name, duration)
        raise
    finally:
        dest[name] = duration


def get_distance_via_iou_and_embeddings(ious_dists, emb_dists, proximity_thresh, appearance_thresh):
    # check if emb_dists < appearance_thresh. Lower emb_dist is better. if not low enough, will use IOU only.
    emb_dists[emb_dists > appearance_thresh] = 1.0
    # # check if iou < proximity_thresh. Lower iou dist is better.
    # ious_dists_mask = ious_dists > proximity_thresh
    # emb_dists[ious_dists_mask] = 1.0
    # print(f"IOUs:\t{ious_dists}")
    # print(f'Embedding dists:\t{emb_dists}')
    return np.minimum(ious_dists, emb_dists)


def get_frame_number(frame_image_name: str) -> int:
    return int(frame_image_name.split('_')[-1].split('.')[0])


def reduce_embeddings_dim_via_pca(point_dict, d):
    all_embeddings = []
    for k in point_dict.keys():
        all_embeddings += point_dict[k]
    temp_p_emb = {}
    pca = PCA(n_components=d).fit(all_embeddings)
    for p, v in point_dict.items():
        temp_p_emb[p] = PCA.transform(pca, v)
    return temp_p_emb


def plot_points_from_dict(point_dict):
    """
    Create a scatter plot of points from a dictionary with different colors for each key.

    Args:
        point_dict (dict): A dictionary where keys are labels and values are lists of 2D numpy arrays.
    """
    for d in [2, 3]:
        temp_p_emb = reduce_embeddings_dim_via_pca(point_dict, d)
        unique_colors = plt.cm.rainbow(np.linspace(0, 1, len(temp_p_emb)))
        if d == 3:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            for i, (key, points) in enumerate(temp_p_emb.items()):
                x, y, z = zip(*points)
                ax.scatter(x, y, z, label=key, c=[unique_colors[i]], s=50)
        else:
            for i, (key, points) in enumerate(temp_p_emb.items()):
                x, y, = zip(*points)
                plt.scatter(x, y, label=key, c=[unique_colors[i]], s=50)

        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend(loc='best')
        plt.title('Scatter Plot of Points by Key')
        plt.show()
