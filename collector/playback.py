# from queue import Queue
# from threading import Thread
#
# import cv2
#
#
# class Playback:
#     def __init__(self, frame=None):
#         self.frame = frame
#         self.stopped = False
#         self.Q = Queue()
#         self.start()
#
#     def start(self):
#         thread = Thread(target=self.show, args=())
#         thread.start()
#
#         return self
#
#     def show(self):
#         while True:
#             if self.more():
#                 cv2.imshow("Video Thread", self.read())
#                 if cv2.waitKey(1) == ord("q"):
#                     self.stopped = True
#
#     def add(self, frame):
#         if not self.Q.full():
#             self.Q.put(frame)
#
#     def read(self):
#         return self.Q.get()
#
#     def more(self):
#         return self.Q.qsize() > 0
#
#     @classmethod
#     def stop(cls):
#         cls.stopped = True
from threading import Thread

import cv2


class Playback:
    """
    Class that continuously shows a frame using a dedicated thread.
    """

    def __init__(self, frame=0):
        self.frame = frame
        self.stopped = False
        self.start()

    def start(self):
        thread = Thread(target=self.show, args=())
        thread.daemon = True
        thread.start()

        return self

    def show(self):
        print('started')
        while True:
            print(self.frame)
            cv2.imshow("Video", self.frame)
            if cv2.waitKey(1) == ord("q"):
                self.stopped = True

    def stop(self):
        self.stopped = True
