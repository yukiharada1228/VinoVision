import logging
import subprocess
import sys
from pathlib import Path

from model import FacialDetectionModel, FacialLandmarkRegressionModel
from openvino.inference_engine import IECore

# ロギングの設定
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logger = logging.getLogger(__name__)
IECORE = IECore()
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = {}
FACE_DETECTION_MODEL = "face-detection-retail-0005"
LANDMARK_REGRESSION_MODEL = "landmarks-regression-retail-0009"
MODELS = [FACE_DETECTION_MODEL, LANDMARK_REGRESSION_MODEL]
for model in MODELS:
    cmd = f"omz_downloader --name {model}"
    model_dir = PROJECT_ROOT / "intel" / model
    model_path = str(model_dir / f"FP32/{model}")
    if not model_dir.exists():
        subprocess.call(cmd.split(" "), cwd=str(PROJECT_ROOT))
    MODEL_PATH[model] = model_path
face_detector = FacialDetectionModel(IECORE, MODEL_PATH[FACE_DETECTION_MODEL])
landmark_regression = FacialLandmarkRegressionModel(
    IECORE, MODEL_PATH[LANDMARK_REGRESSION_MODEL]
)


def landmarks_regress(frame):
    input_frame = face_detector.prepare_frame(frame)
    infer_result = face_detector.infer(input_frame)
    data_array = face_detector.prepare_data(infer_result, frame)
    for data in data_array:
        face_frame, xmin, ymin, xmax, ymax = face_detector.crop(data, frame)
        input_frame = landmark_regression.prepare_frame(face_frame)
        infer_result = landmark_regression.infer(input_frame)
        landmark_regression.draw(infer_result, frame, xmin, ymin, xmax, ymax)
    return frame


if __name__ == "__main__":
    from camera import camera

    @camera
    def camera_landmarks_regress(frame):
        frame = landmarks_regress(frame)
        return frame

    camera_landmarks_regress()
