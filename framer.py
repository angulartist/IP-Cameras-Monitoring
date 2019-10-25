import argparse
import base64
import logging
import logging.config as cfg
import math

import cv2


class FrameHelper(object):

    @staticmethod
    def rescale_frame(frame, scale_percent=50):
        """
        Rescale a frame for the given scale percent.
        :param frame: The frame to rescale.
        :param scale_percent: The scale percent ratio to apply. Default is 50.
        :return: A new rescaled frame.
        """
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)

        return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


class Logger(object):
    def __init__(self, path):
        cfg.fileConfig(path)
        self.logger = logging.getLogger('root')

    def get_logger(self):
        return self.logger


def main():
    # args parser
    parser = argparse.ArgumentParser()
    parser.add_argument('stream', help='static video or video stream to process')
    args = parser.parse_args()

    # init process
    logger = Logger(path='./logging.conf.ini').get_logger()
    stream = cv2.VideoCapture(args.stream)
    global_frame_rate = stream.get(5)
    frame_helper = FrameHelper()

    while stream.isOpened():
        current_frame = stream.get(1)
        has_frames, frame = stream.read()

        if not has_frames:
            break

        if current_frame % math.floor(global_frame_rate) == 0:
            # Rescale image to lower the weight (Vision API requires less than 10MB json body)
            frame_helper.rescale_frame(frame, scale_percent=50)
            _, buffer = cv2.imencode('.jpg', frame)
            base64string = base64.b64encode(buffer)
            logger.info(base64string.decode('utf-8'))
            print('Framed!')

    # Release the stream
    stream.release()


if __name__ == '__main__':
    main()
