from __future__ import absolute_import

import argparse
import threading
from collections import deque
from multiprocessing.pool import ThreadPool

import numpy as np

from collector.stream import StreamHandler
from collector.utils import *


class Framer(object):
    def __init__(self, stream, resolution, frame_rate=10):
        self.mq = PubSubClient()
        # Multiprocessing
        self.num_threads = cv2.getNumberOfCPUs()
        self.pool = ThreadPool(processes=self.num_threads)
        self.pending = deque()
        # Stream
        self.stream = StreamHandler(stream, resolution=resolution, frame_rate=frame_rate)
        self.height, self.width = available_res[resolution].values()
        # ML
        """Uncomment to test object detection"""
        # PROTO_PATH = './ml_processing/model/trained/model.pbtxt'
        # MODEL_PATH = './ml_processing/model/trained/frozen_inference_graph.pb'
        # print("[ML] Loading the model 🥶")
        # self.net = cv2.dnn.readNetFromTensorflow(MODEL_PATH, PROTO_PATH)

    @staticmethod
    def display(frame):
        cv2.imshow('Stream', frame)
        cv2.waitKey(10)

    def add_to_queue(self, frame):
        ext = '.jpg'
        params = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        _, buffer = cv2.imencode(ext, frame, params)
        self.mq.publish(buffer.tobytes())

    def process_frame(self, frame, t0):
        np_frame = np.frombuffer(frame, dtype='uint8') \
            .reshape((self.width, self.height, 3))
        self.add_to_queue(np_frame)

        return np_frame, t0

    def process(self):
        prev = 0

        while True:
            while (len(self.pending) > 0) and (self.pending[0].ready()):
                frame, t0 = self.pending.popleft().get()
                print("Threads: {}".format(threading.active_count()))
                """Uncomment to watch stream"""
                self.display(frame)

            if len(self.pending) < self.num_threads:
                t = clock()
                time_elapsed = time.time() - prev
                frame = self.stream.get_frame()

                if time_elapsed > 1. / self.stream.frame_rate:
                    prev = time.time()
                    task = self.pool.apply_async(self.process_frame, (frame, t))
                    self.pending.append(task)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--stream', help='Video Live Stream URL', type=str)
    parser.add_argument('-r', '--resolution', help='Resolution', type=str)
    args = parser.parse_args()

    framer = Framer(args.stream, args.resolution, frame_rate=10)
    framer.process()
