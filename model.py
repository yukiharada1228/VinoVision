import logging

import cv2 as cv
import numpy as np
from openvino.inference_engine import IECore

from camera import Camera

# ロギングの設定
logger = logging.getLogger(__name__)


class Model(object):
    def __init__(self, ie_core, model_path, device_name="CPU", num_requests=0):
        # モデルの読み込み
        net = ie_core.read_network(model_path + ".xml", model_path + ".bin")

        # モデルをデバイス上で実行するための準備
        self.exec_net = ie_core.load_network(
            network=net, device_name=device_name, num_requests=num_requests
        )

        # 入力名・出力名・入力サイズ・出力サイズを設定
        self.input_name = next(iter(net.input_info))
        self.output_name = next(iter(net.outputs))
        self.input_size = net.input_info[self.input_name].input_data.shape
        self.output_size = (
            self.exec_net.requests[0].output_blobs[self.output_name].buffer.shape
        )

    def prepare_frame(self, frame):
        # 入力サイズにリサイズし、画像のチャンネルを先頭に配置，4次元配列に変換
        _, _, h, w = self.input_size
        input_frame = cv.resize(frame, (h, w)).transpose((2, 0, 1))[np.newaxis]

        # ログ出力
        logger.debug(
            {
                "action": "prepare_frame",
                "input_size": (h, w),
                "input_frame.shape": input_frame.shape,
            }
        )

        return input_frame

    def infer(self, data):
        # 入力データを辞書形式に変換
        input_data = {self.input_name: data}

        # 推論を実行し、推論結果を取得
        infer_result = self.exec_net.infer(input_data)[self.output_name]

        # ログ出力
        logger.debug(
            {
                "action": "infer",
                "input_data.shape": input_data[self.input_name].shape,
                "infer_result.shape": infer_result.shape,
            }
        )

        return infer_result


class FacialDetectionModel(Model):
    def __init__(self, ie_core, model_path, device_name="CPU", num_requests=0):
        super(FacialDetectionModel, self).__init__(
            ie_core=ie_core,
            model_path=model_path,
            device_name=device_name,
            num_requests=num_requests,
        )

    def prepare_data(self, input, frame, confidence=0.5):
        data_array = []
        index_conf = 2
        index_xmin = 3
        index_ymin = 4
        index_xman = 5
        index_ymax = 6
        index_x = 1
        index_y = 0
        for data in np.squeeze(input):
            conf = data[index_conf]
            xmin = max(0, int(data[index_xmin] * frame.shape[index_x]))
            ymin = max(0, int(data[index_ymin] * frame.shape[index_y]))
            xmax = min(
                int(data[index_xman] * frame.shape[index_x]), frame.shape[index_x]
            )
            ymax = min(
                int(data[index_ymax] * frame.shape[index_y]), frame.shape[index_y]
            )
            if conf > confidence:
                area = (xmax - xmin) * (ymax - ymin)
                data = {
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "area": area,
                }
                data_array.append(data)
                data_array.sort(key=lambda face: face["area"], reverse=True)
        logger.debug(
            {
                "action": "prepare_data",
                "input.shape": input.shape,
                "data_array": data_array,
            }
        )
        return data_array

    def draw(self, data_array, frame):
        blue = (255, 0, 0)
        for data in data_array:
            cv.rectangle(
                frame,
                (int(data["xmin"]), int(data["ymin"])),
                (int(data["xmax"]), int(data["ymax"])),
                color=blue,
                thickness=3,
            )
            logger.debug(
                {
                    "action": "draw",
                    "frame.shape": frame.shape,
                }
            )
        return frame

    def crop(self, data, frame):
        xmin = data["xmin"]
        ymin = data["ymin"]
        xmax = data["xmax"]
        ymax = data["ymax"]
        face_frame = frame[ymin:ymax, xmin:xmax]
        logger.debug({"action": "crop", "face_frame.shape": face_frame.shape})
        return face_frame, xmin, ymin, xmax, ymax


class FacialLandmarkRegressionModel(Model):
    def __init__(self, ie_core, model_path, device_name="CPU", num_requests=0):
        super(FacialLandmarkRegressionModel, self).__init__(
            ie_core=ie_core,
            model_path=model_path,
            device_name=device_name,
            num_requests=num_requests,
        )

    def draw(self, infer_result, frame, xmin, ymin, xmax, ymax, max_num=5):
        color_picker = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (0, 255, 255),
            (255, 0, 255),
        ]
        parts = [
            "right_eye",
            "left_eye",
            "nose",
            "right_mouth",
            "left_mouth",
        ]
        width = xmax - xmin
        height = ymax - ymin
        data = np.squeeze(infer_result)
        for index in range(max_num):
            x = int(data[2 * index] * width) + xmin
            y = int(data[2 * index + 1] * height) + ymin
            cv.circle(frame, (x, y), 10, color_picker[index], thickness=-1)
            cv.putText(
                frame,
                str(index),
                (x, y - 10),
                cv.FONT_HERSHEY_PLAIN,
                2,
                color_picker[index],
                1,
                cv.LINE_AA,
            )
            logger.debug({"action": "draw", "part": parts[index], "x": x, "y": y})

    def crop(self, infer_result, frame, xmin, ymin, xmax, ymax):
        parts = [
            "right_eye",
            "left_eye",
            "nose",
            "right_mouth",
            "left_mouth",
        ]
        data_array = []
        width = xmax - xmin
        height = ymax - ymin
        infer_result = np.squeeze(infer_result)
        for index in range(2):
            x = int(infer_result[2 * index] * width) + xmin
            y = int(infer_result[2 * index + 1] * height) + ymin
            data_array.append((x, y))
            logger.debug({"action": "crop", "part": parts[index], "x": x, "y": y})
        return data_array


if __name__ == "__main__":
    import subprocess
    import sys
    from pathlib import Path

    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    DEVICE = 0
    DELAY = 1
    KEYCODE_ESC = 27
    FACE_FRAME_INDEX_X = 1
    FACE_FRAME_INDEX_Y = 0
    RIGHT_EYE_INDEX = 0
    LEFT_EYE_INDEX = 1
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
                data_array = landmark_regression.crop(
                    infer_result, frame, xmin, ymin, xmax, ymax
                )
                for index, data in enumerate(data_array):
                    x, y = data
                    eye_frame = frame[
                        y - face_frame.shape[FACE_FRAME_INDEX_Y] // 10 : y + face_frame.shape[FACE_FRAME_INDEX_Y] // 10,
                        x - face_frame.shape[FACE_FRAME_INDEX_X] // 10 : x + face_frame.shape[FACE_FRAME_INDEX_X] // 10,
                    ]
                    eye_frame = cv.resize(eye_frame, (300, 300))
                    eye_frame = cv.cvtColor(eye_frame, cv.COLOR_BGR2YUV)
                    eye_frame[:, :, 0] = cv.equalizeHist(eye_frame[:, :, 0])
                    eye_frame = cv.cvtColor(eye_frame, cv.COLOR_YUV2BGR)
                    if index == RIGHT_EYE_INDEX:
                        cv.imshow("right_eye_frame", eye_frame)
                    elif index == LEFT_EYE_INDEX:
                        cv.imshow("left_eye_frame", eye_frame)
                landmark_regression.draw(
                    infer_result, frame, xmin, ymin, xmax, ymax, max_num=2
                )
            cv.imshow("frame", frame)
            key = cv.waitKey(DELAY)
            if key == KEYCODE_ESC:
                raise (KeyboardInterrupt)
    except KeyboardInterrupt as ex:
        logger.warning({"ex": ex})
    finally:
        del camera
        cv.destroyAllWindows()
