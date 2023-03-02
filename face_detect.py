import logging
import subprocess
import sys
from pathlib import Path

import cv2 as cv
from openvino.inference_engine import IECore

from camera import Camera
from model import FacialDetectionModel

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logger = logging.getLogger(__name__)


DEVICE = 0
DELAY = 1
KEYCODE_ESC = 27
IECORE = IECore()
CMD_OMZ = "omz_downloader --name face-detection-retail-0005"
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "intel" / "face-detection-retail-0005"
MODEL_PATH = str(MODEL_DIR / "FP16/face-detection-retail-0005")
if not MODEL_DIR.exists():
    subprocess.call(CMD_OMZ.split(" "), cwd=str(PROJECT_ROOT))
camera = Camera(DEVICE)
face_detector = FacialDetectionModel(IECORE, MODEL_PATH)
try:
    while camera.cap.isOpened():
        _, frame = camera.read()
        input_frame = face_detector.prepare_frame(frame)
        infer_result = face_detector.infer(input_frame)
        data_array = face_detector.prepare_data(infer_result, frame)
        face_detector.draw(data_array, frame)
        cv.imshow("frame", frame)
        logger.debug({"frame.shape": frame.shape})
        key = cv.waitKey(DELAY)
        if key == KEYCODE_ESC:
            raise (KeyboardInterrupt)
except KeyboardInterrupt as ex:
    logger.warning({"ex": ex})
finally:
    del camera
    cv.destroyAllWindows()
