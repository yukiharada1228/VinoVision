import logging
import subprocess
import sys
from pathlib import Path

from openvino.inference_engine import IECore

from camera import camera
from model import FacialDetectionModel

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logger = logging.getLogger(__name__)

IECORE = IECore()
CMD_OMZ = "omz_downloader --name face-detection-retail-0005"
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "intel" / "face-detection-retail-0005"
MODEL_PATH = str(MODEL_DIR / "FP32/face-detection-retail-0005")
if not MODEL_DIR.exists():
    subprocess.call(CMD_OMZ.split(" "), cwd=str(PROJECT_ROOT))
face_detector = FacialDetectionModel(IECORE, MODEL_PATH)


def face_detect(frame):
    input_frame = face_detector.prepare_frame(frame)
    infer_result = face_detector.infer(input_frame)
    data_array = face_detector.prepare_data(infer_result, frame)
    face_detector.draw(data_array, frame)
    return frame


@camera
def camera_face_detect(frame):
    frame = face_detect(frame)
    return frame


camera_face_detect()
