"""
Cut a video stream into N frames with a time rotating log file handler.
Convert each frame into its base64 representation to be pushed to the downstream.
"""

from __future__ import absolute_import

import argparse
import threading
from collections import deque
from multiprocessing.pool import ThreadPool

import numpy as np

from collector.stream import StreamHandler
from collector.utils import *


def main():
    # args parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--stream', help='Video Live Stream URL', type=str)
    parser.add_argument('-r', '--resolution', help='Resolution', type=str)
    args = parser.parse_args()

    """Uncomment to test object detection"""
    # PROTO_PATH = './ml_processing/model/trained/model.pbtxt'
    # MODEL_PATH = './ml_processing/model/trained/frozen_inference_graph.pb'
    # print("[ML] Loading the model 🥶")
    # net = cv2.dnn.readNetFromTensorflow(MODEL_PATH, PROTO_PATH)

    # Utils
    pub = PubSubClient()
    # deeper = Deeper(network=net, confidence=0.3)

    thread_n = cv2.getNumberOfCPUs()
    pool = ThreadPool(processes=thread_n)
    pending = deque()

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
    stream = StreamHandler(args.stream, resolution=args.resolution, frame_rate=10)
    height, width = available_res[args.resolution].values()
    prev = 0

    def show_stream(frame):
        cv2.imshow('Stream', frame)
        cv2.waitKey(10)

    def process_frame(frame_as_buffer, t0):
        # Reshape
        frame = np.frombuffer(frame_as_buffer, dtype='uint8') \
            .reshape((width, height, 3))

        # Publish to queue
        _, buffer = cv2.imencode('.jpg', frame, encode_params)
        # pub.publish(buffer.tobytes())

        return frame, t0

    while True:
        while (len(pending) > 0) and (pending[0].ready()):
            frame, t0 = pending.popleft().get()
            print("Threads: {}".format(threading.active_count()))
            """Uncomment to watch stream"""
            show_stream(frame)

        if len(pending) < thread_n:
            t = clock()
            time_elapsed = time.time() - prev
            frame = stream.get_frame()

            if time_elapsed > 1. / stream.frame_rate:
                prev = time.time()
                task = pool.apply_async(process_frame, (frame, t))
                pending.append(task)


if __name__ == '__main__':
    main()
