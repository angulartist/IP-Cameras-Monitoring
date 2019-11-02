import datetime as dt
import time
from queue import Queue
from threading import Thread


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
