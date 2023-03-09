import logging
import subprocess
import sys
from pathlib import Path

from openvino.inference_engine import IECore

from model import EmotionsRecognition, FacialDetectionModel

# ロギングの設定
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logger = logging.getLogger(__name__)
IECORE = IECore()
PROJECT_ROOT = Path(__file__).resolve().parent
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


def emotions_regress(frame):
    input_frame = face_detector.prepare_frame(frame)
    infer_result = face_detector.infer(input_frame)
    data_array = face_detector.prepare_data(infer_result, frame)
    for data in data_array:
        face_frame, xmin, ymin, xmax, ymax = face_detector.crop(data, frame)
        input_frame = emotions_regression.prepare_frame(face_frame)
        infer_result = emotions_regression.infer(input_frame)
        emotions_score = emotions_regression.score(infer_result)
        emotions_regression.draw(xmin, ymin, xmax, ymax, emotions_score, frame)
    return frame


if __name__ == "__main__":
    from camera import camera

    @camera
    def camera_emotions_regress(frame):
        frame = emotions_regress(frame)
        return frame

    camera_emotions_regress()
