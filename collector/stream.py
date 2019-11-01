import subprocess as sp

import streamlink

from collector.utils import available_res


class StreamHandler:
    def __init__(self, url, resolution='720p', frame_rate=30):
        self.stopped = False
        self.url = url
        self.res = resolution
        self.frame_rate = frame_rate
        self.height = 0
        self.width = 0
        self.pipe = None
        self.stream_url = None

        if not self.create_pipe():
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

        self.stream_url = stream.url
        self.pipe = sp.Popen(['ffmpeg', "-i", self.stream_url,
                              "-loglevel", "quiet", "-an",
                              "-f", "image2pipe",
                              "-pix_fmt", "bgr24",
                              "-vcodec", "rawvideo", "-"],
                             stdin=sp.PIPE, stdout=sp.PIPE)

        return True

    def get_frame(self):
        return self.pipe.stdout.read(self.height * self.width * 3)

    @classmethod
    def stop(cls):
        cls.stopped = True
