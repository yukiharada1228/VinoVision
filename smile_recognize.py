import logging
import subprocess
from pathlib import Path

from openvino.inference_engine import IECore

logger = logging.getLogger(__name__)
IECORE = IECore()
PROJECT_ROOT = Path(__file__).resolve().parent
import sys

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


def smile_regress(frame):
    input_frame = face_detector.prepare_frame(frame)
    infer_result = face_detector.infer(input_frame)
    data_array = face_detector.prepare_data(infer_result, frame)
    emotions_score_list = []
    face_frame_list = []
    face_frame = frame.copy()
    for data in data_array:
        face_frame, xmin, ymin, xmax, ymax = face_detector.crop(data, face_frame)
        input_frame = emotions_regression.prepare_frame(face_frame)
        infer_result = emotions_regression.infer(input_frame)
        emotions_score = emotions_regression.score(infer_result)
        emotions_regression.draw(
            xmin, ymin, xmax, ymax, emotions_score, frame, smile_mode=True
        )
        emotions_score_list.append(emotions_score)
        face_frame_list.append(face_frame)
    return emotions_score_list, face_frame_list, frame


if __name__ == "__main__":
    import sys

    from camera import camera

    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    @camera
    def camera_smile_regress(frame):
        _, _, frame = smile_regress(frame)
        return frame

    camera_smile_regress()
