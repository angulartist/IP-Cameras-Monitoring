from __future__ import absolute_import

import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image


class Deeper(object):
    def __init__(self, network, confidence=.25):
        self.network = network
        self.confidence = confidence
        self.classes = {0 : 'background', 1: 'person', 2: 'bicycle',
                        3 : 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
                        7 : 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
                        13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
                        18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
                        24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
                        32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
                        37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
                        41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
                        46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
                        51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
                        56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
                        61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
                        67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
                        75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
                        80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
                        86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier',
                        90: 'toothbrush'}
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def detect(self, base64string):
        assert not isinstance(base64string, type(None)), 'Frame not found! ❌'

        image = cv2.cvtColor(np.array(Image.open(
                BytesIO(base64.b64decode(base64string))
        )), cv2.COLOR_BGR2RGB)

        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True)

        print("[ML] Computing object detections 👻")
        self.network.setInput(blob)
        detections = self.network.forward()

        labels = []
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > self.confidence:
                index = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                label_with_confidence = "{}: {:.2f}%".format(self.classes[index], confidence * 100)

                labels.append('%s' % self.classes[index])

                cv2.rectangle(image,
                              (startX, startY),
                              (endX, endY),
                              self.colors[index], 2)

                y = startY - 15 if startY - 15 > 15 else startY + 15

                cv2.putText(image,
                            label_with_confidence,
                            (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[index], 2)

        # return labels

        """Uncomment to test object detection"""
        cv2.imshow("Output", image)
        cv2.waitKey(1)
