import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
from openvino.inference_engine import IECore

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)
IECORE = IECore()
PROJECT_ROOT = Path(__file__).resolve().parent

import cv2 as cv
from video import Video

sys.path.append(str(PROJECT_ROOT))
from model import EmotionsRecognition, FacialDetectionModel

MODEL_PATH = {}
FACE_DETECTION_MODEL = "face-detection-retail-0005"
EMOTIONS_REGRESSION_MODEL = "emotions-recognition-retail-0003"
MODELS = [FACE_DETECTION_MODEL, EMOTIONS_REGRESSION_MODEL]
for model in MODELS:
    cmd = f"omz_downloader --name {model}"
    model_dir = PROJECT_ROOT / "intel" / model
    model_path = str(model_dir / f"FP32/{model}")
    if not model_dir.exists():
        subprocess.call(cmd.split(" "), cwd=str(PROJECT_ROOT))
    MODEL_PATH[model] = model_path
face_detector = FacialDetectionModel(IECORE, MODEL_PATH[FACE_DETECTION_MODEL])
emotions_regression = EmotionsRecognition(IECORE, MODEL_PATH[EMOTIONS_REGRESSION_MODEL])


def calculate_edge_strength(img):
    img = cv.resize(img, (200, 200))
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    kernel_x = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]], dtype=np.float32)
    kernel_y = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], dtype=np.float32)
    gray_x = cv.filter2D(gray, cv.CV_32F, kernel_x)
    gray_y = cv.filter2D(gray, cv.CV_32F, kernel_y)
    dst = gray_x**2 + gray_y**2
    score = cv.mean(dst)[0]
    logger.debug({"action": "calculate_edge_strength", "score": score})
    return score


def smile_extractor(path):
    def _smile_extractor(process):
        def inner():
            video = Video(path)
            try:
                while video.cap.isOpened():
                    ret, frame = video.read()
                    if not ret:
                        break
                    frame = process(frame)
                    logger.debug({"frame.shape": frame.shape})
            except KeyboardInterrupt as ex:
                logger.warning({"ex": ex})
            finally:
                del video

        return inner

    return _smile_extractor


path = "IMG_6369.MOV"


@smile_extractor(path)
def process(frame):
    input_frame = face_detector.prepare_frame(frame)
    infer_result = face_detector.infer(input_frame)
    data_array = face_detector.prepare_data(infer_result, frame)
    for i, data in enumerate(data_array):
        face_frame, xmin, ymin, xmax, ymax = face_detector.crop(data, frame)
        input_frame = emotions_regression.prepare_frame(face_frame)
        infer_result = emotions_regression.infer(input_frame)
        emotions_score = emotions_regression.score(infer_result)
        emotion = np.argmax(emotions_score)
        if emotion == 1:
            score = calculate_edge_strength(face_frame)
            cv.imwrite(f"img/face_frame/{int(score)}.jpg", face_frame)
            cv.imwrite(f"img/frame/{int(score)}.jpg", frame)

    return frame


process()
