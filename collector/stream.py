import subprocess as sp
import time
from queue import Queue
from threading import Thread

import streamlink

from collector.utils import available_res


class StreamHandler:
    def __init__(self, url, queue_size=512, resolution='720p', frame_rate=30):
        self.stopped = False
        self.url = url
        self.res = resolution
        self.frame_rate = frame_rate
        self.height = 0
        self.width = 0
        self.pipe = None
        self.stream_url = None
        self.Q = Queue(maxsize=queue_size)

        if self.create_pipe():
            self.start_buffer()
        else:
            print('Unable to create stream [pipe]')
            return

    def create_pipe(self):
        try:
            streams = streamlink.streams(self.url)
        except streamlink.exceptions.NoPluginError:
            return False

        if self.res in streams:
            final_res = self.res
        else:
            for key in available_res:
                if key != self.res and key in streams:
                    final_res = key
                    break
            else:
                return False

        self.height = available_res[final_res]["height"]
        self.width = available_res[final_res]["width"]

        stream = streams[final_res]
        self.pipe = sp.Popen([
                'ffmpeg', "-i", stream.url, "-loglevel", "quiet", "-an",
                "-f", "image2pipe", "-pix_fmt", "bgr24", "-vcodec", "rawvideo", "-"
        ], stdin=sp.PIPE, stdout=sp.PIPE)

        return True

    def get_frame(self):
        return self.pipe.stdout.read(self.height * self.width * 3)

    def start_buffer(self):
        thread = Thread(target=self.update_buffer, args=())
        thread.daemon = True
        thread.start()

        return self

    def update_buffer(self):
        prev = 0
        while True:
            time_elapsed = time.time() - prev
            if time_elapsed > 1. / self.frame_rate:
                prev = time.time()
                if not self.Q.full():
                    self.Q.put(self.get_frame())

    def read(self):
        return self.Q.get()

    def more(self):
        return self.Q.qsize() > 0

    @classmethod
    def stop(cls):
        cls.stopped = True
