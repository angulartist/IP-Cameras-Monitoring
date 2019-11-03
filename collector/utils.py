from __future__ import absolute_import

import cv2


def clock():
    return cv2.getTickCount() / cv2.getTickFrequency()


class StatValue:
    def __init__(self, smooth_coef=0.5):
        self.value = None
        self.smooth_coef = smooth_coef

    def update(self, v):
        if self.value is None:
            self.value = v
        else:
            c = self.smooth_coef
            self.value = c * self.value + (1.0 - c) * v


class FrameHelper(object):
    @staticmethod
    def rescale_frame(frame, scale_percent=50):
        """
        Rescale a frame for the given scale percent.
        :param frame: The frame to rescale.
        :param scale_percent: The scale percent ratio to apply. Default is 50.
        :return: A new rescaled frame.
        """

        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)

        return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    @staticmethod
    def resize_frame(frame, width=640, height=480):
        return cv2.resize(frame, (width, height))


resolutions = {
        '360p': (640, 360),
        '720p': (1280, 720)
}
