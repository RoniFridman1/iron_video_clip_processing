from collections import defaultdict
from typing import Iterable, Mapping
import logging
import dataclasses
import enum
import os
import cv2
import shutil
import glob
from imutils import paths
from deepface import DeepFace

# from . import utils
import utils
from ironvideo.wrappers import inference_codeformer

_EMOTION_COMFIDENCE_THRESHOLD = 0 # Emotion score is between 0 and 100.
_CODEFORMER_FIDELITY = 0.5

class Emotion(enum.Enum):
    NONE = 0
    ANGRY = 1
    DISGUST = 2
    FEAR = 3
    HAPPY = 4
    SAD = 5
    SURPRISE = 6
    NEUTRAL = 7


@dataclasses.dataclass
class ImageEmotionTag:
    path: str = ''
    emotion: Emotion = Emotion.NONE
    score: float = float('-inf')


def get_best_image_per_emotion(image_paths: Iterable[str]) -> Mapping[Emotion, ImageEmotionTag]:
    if not image_paths:
        return None
    best_path_per_emotion = defaultdict(ImageEmotionTag)
    for image_path in image_paths:
        actions = ["emotion"]
        result = DeepFace.analyze(image_path, actions=actions, enforce_detection=False, silent=True)
        if result is None:
            logging.warning(f'No emotions for img: {image_path}.')
            continue
        for emotion in Emotion:
            if emotion is Emotion.NONE:
                continue
            emotion_score = result[0]['emotion'][emotion.name.lower()]
            if emotion not in best_path_per_emotion or emotion_score > best_path_per_emotion[emotion].score:
                best_path_per_emotion[emotion] = ImageEmotionTag(path=image_path, emotion=emotion, score=emotion_score)
    return best_path_per_emotion


def get_top_resolution_images_per_person(person_folder_name: str, top: int) -> Iterable[str]:
    # TODO refactor
    images_folder = os.path.join(person_folder_name, utils.ALL_PICS_FOLDER_NAME)
    image_paths = list(paths.list_images(images_folder))
    sorted_images = [(cv2.imread(path), path) for path in image_paths]
    sorted_images = sorted(sorted_images, key=lambda x: x[0].shape[0] * x[0].shape[1], reverse=True)
    return [image[1] for image in sorted_images[:top]]


def get_best_image_paths(person_folder_name):
    top_resoultion_paths = get_top_resolution_images_per_person(person_folder_name, 40)
    best_image_per_emotion = get_best_image_per_emotion(top_resoultion_paths)
    
    # TODO: decide what is the best picture.
    #  As for now we will use the best NEUTRAL picture as the search picture (If there is one).

    best_image_paths = []

    for emotion in [Emotion.NEUTRAL, Emotion.FEAR, Emotion.ANGRY, Emotion.DISGUST, Emotion.HAPPY, Emotion.SAD, Emotion.SURPRISE]:
        if emotion in best_image_per_emotion and best_image_per_emotion[emotion].score > _EMOTION_COMFIDENCE_THRESHOLD:
            path = best_image_per_emotion[emotion].path
            if path not in best_image_paths:
                best_image_paths.append(path)

    return best_image_paths


def restore_cropped_face(image_path: str, output_dir: str, suffix: str, is_cropped_and_aligned=False, *, inference: inference_codeformer.CodeFormerInference) -> None:
    # TODO: refactor CodeFormer to a lib and import normally.
    # TODO: refactor ...

    if is_cropped_and_aligned:
        suffix_new_name = "_has_aligned" + suffix
        has_aligned_parameter = True
    else:
        suffix = suffix+"_00"
        suffix_new_name = suffix
        has_aligned_parameter = False
    inference.run(
        fidelity_weight=_CODEFORMER_FIDELITY,
        has_aligned=has_aligned_parameter,
        input_path=image_path
    )
    restored_face_path = f'results/test_img_{_CODEFORMER_FIDELITY}/restored_faces/'
    restored_default_file_name = f'main_pic{suffix}.png'
    restored_new_name = f'main_pic_restored{suffix_new_name}.png'
    dest_path = os.path.join(output_dir, utils.MAIN_PIC_FOLDER_NAME)
    restored_new_image_path = os.path.join(restored_face_path, restored_new_name)
    try:
        os.rename(os.path.join(restored_face_path, restored_default_file_name), restored_new_image_path)
        shutil.copy2(restored_new_image_path, dest_path)
        shutil.rmtree('results')
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception:
        logging.exception(f'Failed copying restored image')
        

class CodeFormerRestoreFace():
    def __init__(self) -> None:
        # TODO: refactor relative path in CodeFormer/basicsr
        curr_dir = os.getcwd()
        os.chdir('CodeFormer')
        os.system('python basicsr/setup.py develop')
        os.chdir(curr_dir)
        os.system('python CodeFormer/scripts/download_pretrained_models.py all')
        self.inference = inference_codeformer.CodeFormerInference()
    
    def restore_main_pictures(self, video_output_dir: str) -> None:
        for person_dir in os.listdir(video_output_dir):
            person_path = os.path.join(video_output_dir, person_dir)
            for path in glob.glob(os.path.join(person_path, utils.MAIN_PIC_FOLDER_NAME)+"/*.jpg"):
                image_file_name = os.path.basename(path)
                if any([str(int(r*100)) in image_file_name for r in utils.DEFAULT_CROP_MARGIN_RATIO_LIST]):
                    suffix = image_file_name.split("_")[-1].split(".")[0]
                    suffix = f"_{suffix}"
                else:
                    suffix = ''
                best_image_path = os.path.join(person_path, utils.MAIN_PIC_FOLDER_NAME, os.path.basename(path))
                if not os.path.exists(best_image_path):
                    logging.warning(f'Main pic path does not exists: {best_image_path}. Skipping.')
                    continue
                restore_cropped_face(best_image_path, person_path, suffix, inference=self.inference)
                restore_cropped_face(best_image_path, person_path, suffix, is_cropped_and_aligned=True, inference=self.inference)
