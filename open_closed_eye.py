import logging
import subprocess

import cv2 as cv
from camera import camera
from model import (FacialDetectionModel, FacialLandmarkRegressionModel,
                   OpenClosedEyeRegression)
from openvino.inference_engine import IECore

logger = logging.getLogger(__name__)
import sys
from pathlib import Path

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
FACE_FRAME_INDEX_X = 1
FACE_FRAME_INDEX_Y = 0
RIGHT_EYE_INDEX = 0
LEFT_EYE_INDEX = 1
IECORE = IECore()
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = {}
FACE_DETECTION_MODEL = "face-detection-retail-0005"
LANDMARK_REGRESSION_MODEL = "landmarks-regression-retail-0009"
OPEN_CLOSED_EYE_MODEL = "open-closed-eye-0001"
MODELS = [FACE_DETECTION_MODEL, LANDMARK_REGRESSION_MODEL]
for model in MODELS:
    cmd = f"omz_downloader --name {model}"
    model_dir = PROJECT_ROOT / "intel" / model
    model_path = str(model_dir / f"FP32/{model}")
    if not model_dir.exists():
        subprocess.call(cmd.split(" "), cwd=str(PROJECT_ROOT))
    MODEL_PATH[model] = model_path
cmd = f"omz_downloader --name {OPEN_CLOSED_EYE_MODEL}"
model_dir = PROJECT_ROOT / "public" / OPEN_CLOSED_EYE_MODEL
model_path = str(model_dir / f"FP32/{OPEN_CLOSED_EYE_MODEL}")
if not model_dir.exists():
    subprocess.call(cmd.split(" "), cwd=str(PROJECT_ROOT))
    subprocess.call(
        f"omz_converter --name {OPEN_CLOSED_EYE_MODEL}".split(" "),
        cwd=str(PROJECT_ROOT),
    )
MODEL_PATH[OPEN_CLOSED_EYE_MODEL] = model_path
face_detector = FacialDetectionModel(IECORE, MODEL_PATH[FACE_DETECTION_MODEL])
landmark_regression = FacialLandmarkRegressionModel(
    IECORE, MODEL_PATH[LANDMARK_REGRESSION_MODEL]
)
open_closed_eye_regression = OpenClosedEyeRegression(
    IECORE, MODEL_PATH[OPEN_CLOSED_EYE_MODEL]
)


@camera
def open_closed_eye_regress(frame):
    input_frame = face_detector.prepare_frame(frame)
    infer_result = face_detector.infer(input_frame)
    data_array = face_detector.prepare_data(infer_result, frame)
    for index, data in enumerate(data_array):
        face_frame, xmin, ymin, xmax, ymax = face_detector.crop(data, frame)
        input_frame = landmark_regression.prepare_frame(face_frame)
        infer_result = landmark_regression.infer(input_frame)
        data_array = landmark_regression.crop(infer_result, xmin, ymin, xmax, ymax)
        for index, data in enumerate(data_array):
            x, y = data
            eye_frame = frame[
                y
                - face_frame.shape[FACE_FRAME_INDEX_Y] // 5 : y
                + face_frame.shape[FACE_FRAME_INDEX_Y] // 5,
                x
                - face_frame.shape[FACE_FRAME_INDEX_X] // 5 : x
                + face_frame.shape[FACE_FRAME_INDEX_X] // 5,
            ]
            input_frame = open_closed_eye_regression.prepare_frame(eye_frame)
            eye_infer_result = open_closed_eye_regression.infer(input_frame)
            eye_infer_result = open_closed_eye_regression.prepare_data(eye_infer_result)
            eye_frame = cv.resize(eye_frame, (300, 300))
            cv.putText(
                eye_frame,
                eye_infer_result,
                (100, 100),
                cv.FONT_HERSHEY_PLAIN,
                2,
                (0, 255, 255) if eye_infer_result == "open" else (255, 0, 0),
                1,
                cv.LINE_AA,
            )
            if index == RIGHT_EYE_INDEX:
                cv.imshow("right_eye_frame", eye_frame)
            elif index == LEFT_EYE_INDEX:
                cv.imshow("left_eye_frame", eye_frame)
        landmark_regression.draw(infer_result, frame, xmin, ymin, xmax, ymax, max_num=2)
    return frame


open_closed_eye_regress()
