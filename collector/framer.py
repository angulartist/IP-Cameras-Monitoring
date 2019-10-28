"""
Cut a video stream into N frames with a time rotating log file handler.
Convert each frame into its base64 representation to be pushed to the downstream.
"""

from __future__ import absolute_import

import argparse
from collections import deque
from multiprocessing.pool import ThreadPool

from collector.utils import *


def main():
    # args parser
    parser = argparse.ArgumentParser()
    parser.add_argument('stream', help='static video or video stream to process')
    args = parser.parse_args()

    """Uncomment to test object detection"""
    # PROTO_PATH = './ml_processing/model/trained/model.pbtxt'
    # MODEL_PATH = './ml_processing/model/trained/frozen_inference_graph.pb'
    # print("[ML] Loading the model 🥶")
    # net = cv2.dnn.readNetFromTensorflow(MODEL_PATH, PROTO_PATH)

    pub_sub_client = PubSubClient()
    frame_helper = FrameHelper()

    thread_n = cv2.getNumberOfCPUs()
    pool = ThreadPool(processes=thread_n)
    pending = deque()

    frame_interval = StatValue()
    last_frame_time = clock()

    stream = cv2.VideoCapture(args.stream)

    def process_frame(current_frame, t0):
        current_frame = frame_helper.rescale_frame(current_frame, scale_percent=40)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        _, buffer = cv2.imencode('.jpg', current_frame, encode_param)
        # print('Will publish frame...')
        pub_sub_client.publish(buffer.tobytes())

        return current_frame, t0

    while stream.isOpened():
        while len(pending) > 0 and pending[0].ready():
            pending.popleft().get()
        if len(pending) < thread_n:
            has_frames, frame = stream.read()

            if not has_frames:
                break

            t = clock()
            frame_interval.update(t - last_frame_time)
            last_frame_time = t

            task = pool.apply_async(process_frame, (frame.copy(), t))
            pending.append(task)
    # Release the stream
    stream.release()


if __name__ == '__main__':
    main()
