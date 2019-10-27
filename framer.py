"""
Cut a video stream into N frames with a time rotating log file handler.
Convert each frame into its base64 representation to be pushed to the downstream.
"""

from __future__ import absolute_import

import argparse
import logging
import logging.config as cfg
import time

import cv2
from google.cloud import pubsub_v1


class PubSub(object):
    def __init__(self, project_id='alert-shape-256811', topic_name='ml-flow'):
        self.publisher = pubsub_v1.PublisherClient(
                pubsub_v1.types.BatchSettings(max_latency=2)
        )
        self.topic_path = self.publisher.topic_path(project_id, topic_name)

    def publish(self, frame_as_bytes):
        future = self.publisher.publish(self.topic_path, data=frame_as_bytes)
        print(future.result())
        print('Published messages with batch settings.')


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
    pub_sub = PubSub()
    # logger = Logger(path='./logging.conf.ini').get_logger()
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
            pub_sub.publish(buffer.tobytes())
            print(stream.get(cv2.CAP_PROP_POS_MSEC) / 1000)
            time.sleep(0.2)

            # Deeper(net, confidence=.3).detect(base64string)

        frame_position += 1

    # Release the stream
    stream.release()


if __name__ == '__main__':
    main()
