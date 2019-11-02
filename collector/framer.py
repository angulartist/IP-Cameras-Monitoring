from __future__ import absolute_import

import argparse
import threading
from collections import deque
from multiprocessing.pool import ThreadPool

import numpy as np

from collector.pubsubclient import PubSubClient
from collector.stream import StreamHandler
from collector.utils import *
from ml_processing.deeper import Deeper


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
        self.PROTO_PATH = './ml_processing/model/trained/model.pbtxt'
        self.MODEL_PATH = './ml_processing/model/trained/frozen_inference_graph.pb'

    def visualize(self, frames):
        print("[ML] Loading the model 🥶")
        net = cv2.dnn.readNetFromTensorflow(self.MODEL_PATH, self.PROTO_PATH)
        deeper = Deeper(network=net, confidence=0.3)
        processed_frames = deeper.detect(frames)
        for frame in processed_frames:
            self.display(frame)

    @staticmethod
    def display(frame):
        cv2.imshow('Stream', frame)
        cv2.waitKey(10)

    def process_frame(self, frame, t0):
        np_frame = np.frombuffer(frame, dtype='uint8') \
            .reshape((self.width, self.height, 3))
        ext = '.jpg'
        params = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        _, buffer = cv2.imencode(ext, np_frame, params)
        self.mq.add_to_queue(buffer)

        return np_frame, t0

    def process(self):
        # frames = []
        while True:
            while (len(self.pending) > 0) and (self.pending[0].ready()):
                frame, _ = self.pending.popleft().get()
                print("Threads: {}".format(threading.active_count()))
                """Uncomment to watch stream"""
                self.display(frame)
                # if len(frames) >= 10:
                #     self.visualize(frames)
                #     frames = []
                # else:
                #     frames.append(frame)

            if self.stream.more and (len(self.pending) < self.num_threads):
                if self.stream.more():
                    t = clock()
                    frame = self.stream.read()
                    task = self.pool.apply_async(self.process_frame, (frame, t))
                    self.pending.append(task)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--stream', help='Video Live Stream URL', type=str)
    parser.add_argument('-r', '--resolution', help='Resolution', type=str)
    args = parser.parse_args()
    # --- #
    framer = Framer(args.stream, args.resolution, frame_rate=30)
    framer.process()
