import logging
import subprocess
import sys
from pathlib import Path

import cv2 as cv
from openvino.inference_engine import IECore

from camera import Camera
from model import FacialDetectionModel, FacialLandmarkRegressionModel

# ロギングの設定
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logger = logging.getLogger(__name__)
DEVICE = 0
DELAY = 1
KEYCODE_ESC = 27
IECORE = IECore()
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = {}
FACE_DETECTION_MODEL = "face-detection-retail-0005"
LANDMARK_REGRESSION_MODEL = "landmarks-regression-retail-0009"
for model in [FACE_DETECTION_MODEL, LANDMARK_REGRESSION_MODEL]:
    cmd = f"omz_downloader --name {model}"
    model_dir = PROJECT_ROOT / "intel" / model
    model_path = str(model_dir / f"FP16/{model}")
    if not model_dir.exists():
        subprocess.call(cmd.split(" "), cwd=str(PROJECT_ROOT))
    MODEL_PATH[model] = model_path
camera = Camera(DEVICE)
face_detector = FacialDetectionModel(IECORE, MODEL_PATH[FACE_DETECTION_MODEL])
landmark_regression = FacialLandmarkRegressionModel(
    IECORE, MODEL_PATH[LANDMARK_REGRESSION_MODEL]
)
try:
    while camera.cap.isOpened():
        _, frame = camera.read()
        input_frame = face_detector.prepare_frame(frame)
        infer_result = face_detector.infer(input_frame)
        data_array = face_detector.prepare_data(infer_result, frame)
        for index, data in enumerate(data_array):
            face_frame, xmin, ymin, xmax, ymax = face_detector.crop(data, frame)
            input_frame = landmark_regression.prepare_frame(face_frame)
            infer_result = landmark_regression.infer(input_frame)
            landmark_frame = landmark_regression.draw(
                infer_result, frame, xmin, ymin, xmax, ymax
            )
        cv.imshow("landmark_frame", landmark_frame)
        key = cv.waitKey(DELAY)
        if key == KEYCODE_ESC:
            raise (KeyboardInterrupt)
except KeyboardInterrupt as ex:
    logger.warning({"ex": ex})
finally:
    del camera
    cv.destroyAllWindows()
