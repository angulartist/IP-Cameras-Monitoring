import subprocess as sp
from queue import Queue
from threading import Thread

import streamlink


class TwitchStreamHandler:
    def __init__(self, twitch_url, queue_size=128, resolution='720p', n_frame=30):
        self.stopped = False
        self.twitch_url = twitch_url
        self.res = resolution
        self.n_frame = n_frame
        self.height = 0
        self.width = 0
        self.pipe = None
        self.stream_url = None

        self.Q = Queue(maxsize=queue_size)
        checkIfStreamsWorks = self.create_pipe()

        if checkIfStreamsWorks:
            self.start_buffer()

    def create_pipe(self):
        try:
            streams = streamlink.streams(self.twitch_url)
        except streamlink.exceptions.NoPluginError:
            return False

        resolutions = {
                '360p': {
                        "height": 640, "width": 360
                },
                '720p': {
                        "height": 1280, "width": 720
                }
        }

        if self.res in streams:
            final_res = self.res
        else:
            for key in resolutions:
                if key != self.res and key in streams:
                    final_res = key
                    break
            else:
                return False

        self.height = resolutions[final_res]["height"]
        self.width = resolutions[final_res]["width"]
        stream = streams[final_res]

        self.stream_url = stream.url
        self.pipe = sp.Popen(['ffmpeg', "-i", self.stream_url,
                              "-loglevel", "quiet", "-an",
                              "-f", "image2pipe",
                              "-pix_fmt", "bgr24",
                              "-vcodec", "rawvideo", "-"],
                             stdin=sp.PIPE, stdout=sp.PIPE)

        return True

    def start_buffer(self):
        thread = Thread(target=self.update_buffer, args=())
        thread.daemon = True
        thread.start()

        return self

    def update_buffer(self):
        iter_frames = 0

        while True:
            if iter_frames % self.n_frame == 0:
                bfr = self.pipe.stdout.read(
                        self.height * self.width * 3)

                if not self.Q.full():
                    self.Q.put(bfr)
                    iter_frames += 1
                else:
                    iter_frames += 1
                    continue
            else:
                iter_frames += 1
                continue

    def read(self):
        return self.Q.get()

    def more(self):
        return self.Q.qsize() > 0

    def get_resolution(self):
        return self.height, self.width

    @classmethod
    def stop(cls):
        cls.stopped = True
