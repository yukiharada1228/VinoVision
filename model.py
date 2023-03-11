import logging
import math

import cv2 as cv
import numpy as np
from openvino.inference_engine import IECore

logger = logging.getLogger(__name__)


class Model(object):
    def __init__(self, ie_core, model_path, device_name="CPU", num_requests=0):
        net = ie_core.read_network(model_path + ".xml", model_path + ".bin")

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

        logger.debug(
            {
                "action": "prepare_frame",
                "input_size": (h, w),
                "input_frame.shape": input_frame.shape,
            }
        )

        return input_frame

    def infer(self, data):
        input_data = {self.input_name: data}

        infer_result = self.exec_net.infer(input_data)[self.output_name]

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

            # フレームからはみ出す座標を避けるため、xmin/yminは0未満にならないようにします
            xmin = max(0, int(data[index_xmin] * frame.shape[index_x]))
            ymin = max(0, int(data[index_ymin] * frame.shape[index_y]))

            # フレーム内に収まるよう、xmax/ymaxをframeの幅/高さの範囲内に制限します
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

                # 検出されたオブジェクトを面積の大きい順にソートし、最も大きいオブジェクトから処理するようにします
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
        """
        このメソッドは、1つの顔の情報だけを処理することで、余分な情報を保持する必要がなくなります。
        そのため、メモリ使用量を削減できます。また、data_arrayのような一時的なリストを使用する必要がないため、
        リストの生成にかかる時間や、メモリ使用量も削減できます。

        Args:
            data: 1つの顔のxmin, ymin, xmax, ymaxが含まれるdict。
            frame: フレームの画像データ

        Returns:
            face_frame: 顔領域の画像データ
            xmin: 顔領域のxmin
            ymin: 顔領域のymin
            xmax: 顔領域のxmax
            ymax: 顔領域のymax
        """
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
        # 領域の幅と高さを計算
        face_width = xmax - xmin
        face_height = ymax - ymin

        # 推論結果を取得し、各ランドマークの位置を計算
        landmark_coords = np.squeeze(infer_result)
        for index in range(max_num):
            # 推論結果から各ランドマークの位置を取得する。
            # 各ランドマークの座標は0.0〜1.0で表されているので、
            # フレーム内の実際の位置に変換するために、
            # フレームの領域幅と高さにそれぞれ掛けてから、
            # フレームのxmin、yminの値を足すことで実際の座標を計算する。
            INDEX_X = 2 * index
            INDEX_Y = 2 * index + 1
            landmark_x = int(landmark_coords[INDEX_X] * face_width) + xmin
            landmark_y = int(landmark_coords[INDEX_Y] * face_height) + ymin

            # 画像上にランドマークを描画
            cv.circle(
                frame, (landmark_x, landmark_y), 10, color_picker[index], thickness=-1
            )
            cv.putText(
                frame,
                str(index),
                (landmark_x, landmark_y - 10),
                cv.FONT_HERSHEY_PLAIN,
                2,
                color_picker[index],
                1,
                cv.LINE_AA,
            )

            logger.debug(
                {
                    "action": "draw",
                    "part": parts[index],
                    "x": landmark_x,
                    "y": landmark_y,
                }
            )

    def crop(self, infer_result, xmin, ymin, xmax, ymax):
        parts = [
            "right_eye",
            "left_eye",
            "nose",
            "right_mouth",
            "left_mouth",
        ]
        landmarks = []
        face_width = xmax - xmin
        face_height = ymax - ymin
        landmark_coords = np.squeeze(infer_result)
        for index in range(len(parts)):
            INDEX_X = 2 * index
            INDEX_Y = 2 * index + 1
            landmark_x = int(landmark_coords[INDEX_X] * face_width) + xmin
            landmark_y = int(landmark_coords[INDEX_Y] * face_height) + ymin
            landmarks.append((landmark_x, landmark_y))
            logger.debug(
                {
                    "action": "crop",
                    "part": parts[index],
                    "landmark_x": landmark_x,
                    "landmark_y": landmark_y,
                }
            )
        # ランドマークの位置座標のリストを返す
        return landmarks


class OpenClosedEyeRegression(Model):
    def __init__(self, ie_core, model_path, device_name="CPU", num_requests=0):
        super(OpenClosedEyeRegression, self).__init__(
            ie_core=ie_core,
            model_path=model_path,
            device_name=device_name,
            num_requests=num_requests,
        )

    def prepare_data(self, output):
        if output[0][0] > output[0][1]:
            return "closed"
        else:
            return "open"


class GenderRecognize(Model):
    def __init__(self, ie_core, model_path, device_name="CPU", num_requests=0):
        super(GenderRecognize, self).__init__(
            ie_core=ie_core,
            model_path=model_path,
            device_name=device_name,
            num_requests=num_requests,
        )

    def infer(self, data):
        input_data = {self.input_name: data}

        infer_result = self.exec_net.infer(input_data)["prob"]

        logger.debug(
            {
                "action": "infer",
                "input_data.shape": input_data[self.input_name].shape,
                "infer_result.shape": infer_result.shape,
            }
        )

        return infer_result

    def prepare_data(self, output):
        output = np.squeeze(output)
        gender = "Female" if output[1] < output[0] else "Male"
        return gender


class EmotionsRecognition(Model):
    def __init__(self, ie_core, model_path, device_name="CPU", num_requests=0):
        super(EmotionsRecognition, self).__init__(
            ie_core=ie_core,
            model_path=model_path,
            device_name=device_name,
            num_requests=num_requests,
        )

    def score(self, infer_result):
        emotions_score = np.squeeze(infer_result)
        return emotions_score

    def draw(self, xmin, ymin, xmax, ymax, emotions_score, frame, smile_mode=False):
        color_picker = [
            (255, 0, 0),
            (0, 255, 255),
            (0, 255, 0),
            (255, 0, 255),
            (0, 0, 255),
        ]
        NEUTRAL_INDEX = 0
        SMILE_INDEX = 1
        emotion = np.argmax(emotions_score)
        if emotion == 1:
            color = color_picker[SMILE_INDEX]
        elif smile_mode:
            color = color_picker[NEUTRAL_INDEX]
        else:
            color = color_picker[emotion]
        cv.rectangle(
            frame,
            (int(xmin), int(ymin)),
            (int(xmax), int(ymax)),
            color=color,
            thickness=3,
        )
        logger.debug(
            {
                "action": "draw",
                "frame.shape": frame.shape,
            }
        )
        return frame


if __name__ == "__main__":
    import logging
    import subprocess
    import sys
    from pathlib import Path

    from camera import camera
    from model import EmotionsRecognition, FacialDetectionModel
    from openvino.inference_engine import IECore

    # ロギングの設定
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger(__name__)
    IECORE = IECore()
    PROJECT_ROOT = Path(__file__).resolve().parent
    MODEL_PATH = {}
    FACE_DETECTION_MODEL = "face-detection-retail-0005"
    LANDMARK_REGRESSION_MODEL = "landmarks-regression-retail-0009"
    AGE_GENDER_RECOGNITION_MODEL = "age-gender-recognition-retail-0013"
    MODELS = [
        FACE_DETECTION_MODEL,
        LANDMARK_REGRESSION_MODEL,
        AGE_GENDER_RECOGNITION_MODEL,
    ]
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
    gender_recognize = GenderRecognize(IECORE, MODEL_PATH[AGE_GENDER_RECOGNITION_MODEL])

    @camera
    def process(frame):
        input_frame = face_detector.prepare_frame(frame)
        infer_result = face_detector.infer(input_frame)
        data_array = face_detector.prepare_data(infer_result, frame)
        for data in data_array:
            face_frame, xmin, ymin, xmax, ymax = face_detector.crop(data, frame)
            input_frame = landmark_regression.prepare_frame(face_frame)
            infer_result = landmark_regression.infer(input_frame)

            landmarks = np.squeeze(infer_result)
            left_eye_x = landmarks[0]
            left_eye_y = landmarks[1]
            right_eye_x = landmarks[2]
            right_eye_y = landmarks[3]
            center_x = (left_eye_x + right_eye_x) / 2
            center_y = (left_eye_y + right_eye_y) / 2
            angle = (
                math.atan2(right_eye_y - left_eye_y, right_eye_x - left_eye_x)
                * 180
                / math.pi
            )
            rows, cols, _ = face_frame.shape
            M = cv.getRotationMatrix2D((center_x, center_y), angle, 1)
            face_frame_rotated = cv.warpAffine(face_frame, M, (cols, rows))
            cv.imshow("face_frame_rotated", face_frame_rotated)

            input_frame = gender_recognize.prepare_frame(face_frame_rotated)
            infer_result = gender_recognize.infer(input_frame)
            gender = gender_recognize.prepare_data(infer_result)
            cv.putText(
                frame,
                gender,
                (100, 100),
                cv.FONT_HERSHEY_PLAIN,
                2,
                (0, 255, 255) if gender == "Female" else (255, 0, 0),
                1,
                cv.LINE_AA,
            )
            logger.info(gender)
        return frame

    process()
