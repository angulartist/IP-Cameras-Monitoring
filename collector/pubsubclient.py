import datetime as dt
import time
from queue import Queue
from threading import Thread

import cv2


class PubSubClient(object):
    def __init__(self, project_id='alert-shape-256811', topic_name='ml-flow'):
        from google.cloud import pubsub

        # Used for batching frames
        settings = pubsub.types.BatchSettings(
                max_messages=30,
                max_latency=2
        )
        # Publisher
        self.publisher = pubsub.PublisherClient(settings)
        self.topic_path = self.publisher.topic_path(project_id, topic_name)
        self.Q = Queue()
        self.start()
        # Subscriber
        self.subscriber = pubsub.SubscriberClient()
        self.subscription_path = self.subscriber.subscription_path(project_id, topic_name)

    def start(self):
        thread = Thread(target=self.publish, args=())
        thread.daemon = True
        thread.start()

        return self

    def publish(self):
        while True:
            if self.more():
                frame = self.read()
                rfc_timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
                self.publisher.publish(self.topic_path, data=frame, timestamp=str(rfc_timestamp))
                print('Published at {}'.format(rfc_timestamp))

            time.sleep(.25)

    def add(self, frame):
        if not self.Q.full():
            ext = '.jpg'
            params = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            _, buffer = cv2.imencode(ext, frame, params)
            self.Q.put_nowait(buffer.tobytes())

    def read(self):
        return self.Q.get_nowait()

    def more(self):
        return self.Q.qsize() > 0
