"""
Cut a video stream into N frames with a time rotating log file handler.
Convert each frame into its base64 representation to be pushed to the downstream.
"""

from __future__ import absolute_import

import argparse
import base64
import logging
import logging.config as cfg

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
        assert not isinstance(frame, type(None)), 'Frame not found! ❌'

        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)

        return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    @staticmethod
    def resize_frame(frame, width=640, height=480):
        assert not isinstance(frame, type(None)), 'Frame not found! ❌'

        return cv2.resize(frame, (width, height))


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

    """Uncomment to test object detection"""
    # PROTO_PATH = './ml-model/proto.pbtxt'
    # MODEL_PATH = './ml-model/frozen_inference_graph.pb'
    # logging.info("[ML] Loading the model 🥶")
    # net = cv2.dnn.readNetFromTensorflow(MODEL_PATH, PROTO_PATH)

    # init process
    logger = Logger(path='./logging.conf.ini').get_logger()
    stream = cv2.VideoCapture(args.stream)
    step = 5
    frame_position = 1
    while stream.isOpened():
        has_frames, frame = stream.read()

        if not has_frames:
            break

        if frame_position % step == 0:
            assert not isinstance(frame, type(None)), 'Frame not found! ❌'

            _, buffer = cv2.imencode('.jpg', frame)
            base64string = base64 \
                .b64encode(buffer) \
                .decode('utf-8')
            logger.info(base64string)
            print(stream.get(cv2.CAP_PROP_POS_MSEC) / 1000)

            # Deeper(net, confidence=.3).detect(base64string)

        frame_position += 1

    # Release the stream
    stream.release()


if __name__ == '__main__':
    main()
