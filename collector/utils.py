from __future__ import absolute_import

import datetime as dt
import time
from queue import Queue
from threading import Thread

import cv2


def clock():
    return cv2.getTickCount() / cv2.getTickFrequency()


class StatValue:
    def __init__(self, smooth_coef=0.5):
        self.value = None
        self.smooth_coef = smooth_coef

    def update(self, v):
        if self.value is None:
            self.value = v
        else:
            c = self.smooth_coef
            self.value = c * self.value + (1.0 - c) * v


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


class PubSubClient(object):
    def __init__(self, project_id='alert-shape-256811', topic_name='ml-flow'):
        from google.cloud import pubsub

        # Used for batching frames
        settings = pubsub.types.BatchSettings(
                max_messages=30,
                max_latency=2,
                max_bytes=1024
        )
        # Publisher
        self.publisher = pubsub.PublisherClient()
        self.topic_path = self.publisher.topic_path(project_id, topic_name)
        self.Q = Queue()
        self.start_queue()
        # Subscriber
        self.subscriber = pubsub.SubscriberClient()
        self.subscription_path = self.subscriber.subscription_path(project_id, topic_name)

    def start_queue(self):
        thread = Thread(target=self.publish, args=())
        thread.daemon = True
        thread.start()

    def publish(self):
        while True:
            if self.more():
                frame = self.read()
                rfc_timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
                self.publisher.publish(self.topic_path, data=frame, timestamp=str(rfc_timestamp))
                print('Published at {}'.format(rfc_timestamp))

    def add_to_queue(self, buffer):
        if not self.Q.full():
            self.Q.put(buffer.tobytes())

    def receive(self):
        def callback(message):
            message.ack()

        self.subscriber.subscribe(self.subscription_path, callback=callback)
        print('Listening for messages on {}'.format(self.subscription_path))
        while True:
            time.sleep(60)

    def read(self):
        return self.Q.get()

    def more(self):
        return self.Q.qsize() > 0


available_res = {
        '360p': {
                "height": 640, "width": 360
        },
        '720p': {
                "height": 1280, "width": 720
        }
}
