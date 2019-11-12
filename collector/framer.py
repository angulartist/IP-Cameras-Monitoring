from __future__ import absolute_import

import argparse
import threading
from collections import deque
from multiprocessing.pool import ThreadPool

import numpy as np

from collector.pubsubclient import PubSubClient
from collector.stream import StreamHandler
from collector.utils import *
from ml_processing.detector import Detector


class Framer(object):
    def __init__(self, stream, resolution, frame_rate=10):
        self.mq = PubSubClient()
        # Multithreading
        self.num_threads = cv2.getNumberOfCPUs()
        self.pool = ThreadPool(processes=self.num_threads)
        self.pending = deque()
        # Stream
        self.stream = StreamHandler(stream, resolution=resolution, frame_rate=frame_rate)
        self.height, self.width = resolutions[resolution]
        # TODO: Add confidence + draw label
        self.detector = Detector()

    @staticmethod
    def display(frame) -> None:
        cv2.imshow('Stream', frame)
        cv2.waitKey(5)

    def process_frame(self, frame_timestamp, _t):
        frame, timestamp = frame_timestamp
        np_frame = np.frombuffer(frame, dtype='uint8') \
            .reshape((self.width, self.height, 3))

        # processed_frame = self.deeper.detect(np_frame)
        # cv2.imwrite("records/frame-{}.jpg".format(timestamp), processed_frame)
        # print(timestamp)

        # Send to mq
        ext = '.jpg'
        params = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        _, buffer = cv2.imencode(ext, np_frame, params)
        # self.mq.add(buffer)

        return np_frame

    def process(self) -> None:
        while True:
            while (len(self.pending) > 0) and (self.pending[0].ready()):
                np_frame = self.pending.popleft().get()
                print("Threads: {}".format(threading.active_count()))
                """Uncomment to watch stream"""
                # self.display(np_frame)
                img = self.detector.detect(np_frame)
                self.display(img)

            if self.stream.more and (len(self.pending) < self.num_threads):
                if self.stream.more():
                    t = clock()
                    frame_timestamp = self.stream.read()
                    task = self.pool.apply_async(self.process_frame, (frame_timestamp, t))
                    self.pending.append(task)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--stream', help='Video Live Stream URL', type=str)
    parser.add_argument('-r', '--resolution', help='Resolution', type=str)
    args = parser.parse_args()
    # --- #
    framer = Framer(args.stream, args.resolution, frame_rate=2)
    framer.process()
