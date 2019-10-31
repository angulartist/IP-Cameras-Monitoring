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
    frame_helper = FrameHelper()
    # deeper = Deeper(network=net, confidence=0.3)

    thread_n = cv2.getNumberOfCPUs()
    pool = ThreadPool(processes=thread_n)
    pending = deque()

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 80]

    stream = StreamHandler(args.stream, resolution=args.resolution)
    height, width = res[args.resolution].values()

    def show_stream(_frame):
        cv2.imshow('Stream', _frame)
        cv2.waitKey(5)

    def process_frame(_frame_as_buffer, _t0):
        _frame = np.frombuffer(_frame_as_buffer, dtype='uint8') \
            .reshape((width, height, 3))

        _, buffer = cv2.imencode('.jpg', _frame, encode_params)
        # print('Will publish frame...')
        pub.publish(buffer.tobytes())

        return _frame, _t0

    while True:
        while len(pending) > 0 and pending[0].ready():
            frame, t0 = pending.popleft().get()
            print("Threads:  %s - %s - %s" % (
                    threading.active_count(), t0, threading.current_thread()))
            """Uncomment to watch stream"""
            # show_stream(frame)

        if len(pending) < thread_n:
            if stream.more():
                t = clock()
                frame_as_buffer = stream.read()
                task = pool.apply_async(process_frame, (frame_as_buffer, t))
                pending.append(task)
            else:
                continue
    # Release the stream
    stream.stop()


if __name__ == '__main__':
    main()
