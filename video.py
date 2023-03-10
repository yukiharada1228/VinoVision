import logging

import cv2 as cv

logger = logging.getLogger(__name__)


class Video(object):
    def __init__(self, path):
        self.cap = cv.VideoCapture(path)
        if not self.cap.isOpened():
            logger.error("動画ファイルを開けませんでした。path: %s", path)
            raise RuntimeError("動画ファイルを開けませんでした。path: %s" % path)
        else:
            logger.debug("動画ファイルを正常に開きました。path: %s", path)

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            logger.warning("動画ファイルからフレームを読み取れませんでした。")
        return ret, frame

    def release(self):
        self.cap.release()
        logger.debug("動画ファイルを正常に解放しました。")

    def __del__(self):
        if hasattr(self, "cap") and self.cap.isOpened():
            self.release()


def video(path):
    def _video(process):
        def inner():
            DELAY = 1
            KEYCODE_ESC = 27
            video = Video(path)
            try:
                while video.cap.isOpened():
                    ret, frame = video.read()
                    if not ret:
                        break
                    frame = process(frame)
                    cv.imshow("frame", frame)
                    logger.debug({"frame.shape": frame.shape})
                    key = cv.waitKey(DELAY)
                    if key == KEYCODE_ESC:
                        raise (KeyboardInterrupt)
            except KeyboardInterrupt as ex:
                logger.warning({"ex": ex})
            finally:
                del video
                cv.destroyAllWindows()

        return inner

    return _video


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    path = "IMG_6369.MOV"

    @video(path)
    def process(frame):
        frame = cv.resize(frame, None, None, 0.5, 0.5)
        logger.debug({"action": "process"})
        return frame

    process()
