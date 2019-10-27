"""
Cut a video stream into N frames with a time rotating log file handler.
Convert each frame into its base64 representation to be pushed to the downstream.
"""

from __future__ import absolute_import

import argparse
import logging
import time

import cv2
from google.cloud import pubsub


class PubSubClient(object):
    def __init__(self, project='alert-shape-256811', topic='ml-flow'):
        settings = pubsub.types.BatchSettings(
                max_messages=10,
                max_latency=5,
        )
        self.publisher = pubsub.PublisherClient(batch_settings=settings)
        self.topic_path = self.publisher.topic_path(project, topic)

    def publish(self, frame_as_bytes):
        future = self.publisher.publish(self.topic_path, data=frame_as_bytes)
        logging.info('Published id: %s', future.result())


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

    @staticmethod
    def resize_frame(frame, width=640, height=480):
        return cv2.resize(frame, (width, height))


def main():
    # args parser
    parser = argparse.ArgumentParser()
    parser.add_argument('stream', help='static video or video stream to process')
    args = parser.parse_args()

    """Uncomment to test object detection"""
    # PROTO_PATH = './ml_processing/model/trained/model.pbtxt'
    # MODEL_PATH = './ml_processing/model/trained/frozen_inference_graph.pb'
    # logging.info("[ML] Loading the model 🥶")
    # net = cv2.dnn.readNetFromTensorflow(MODEL_PATH, PROTO_PATH)

    pub_sub_client = PubSubClient()
    frame_helper = FrameHelper()
    stream = cv2.VideoCapture(args.stream)

    time.sleep(5.0)
    while stream.isOpened():
        has_frames, frame = stream.read()

        if not has_frames:
            break

        frame = frame_helper.rescale_frame(frame, scale_percent=40)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        logging.info('Will publish frame...')
        pub_sub_client.publish(buffer.tobytes())

        """Uncomment to visualize labels and boxes"""
        # labels = Deeper(net, confidence=.3).detect(frame)

        # Avoid CPU intensive work
        time.sleep(0.2)
    # Release the stream
    stream.release()


if __name__ == '__main__':
    main()
