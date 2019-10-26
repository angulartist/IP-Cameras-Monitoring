import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image


class Deeper(object):
    def __init__(self, prototype, model, confidence=.25):
        self.prototype = prototype
        self.model = model
        self.confidence = confidence

        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]
        self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))

    def detect(self, base64string):
        assert not isinstance(base64string, type(None)), 'Frame not found!'

        image = cv2.cvtColor(np.array(Image.open(BytesIO(base64.b64decode(base64string)))),
                             cv2.COLOR_BGR2RGB)

        print("[INFO] loading model...")
        net = cv2.dnn.readNetFromCaffe(self.prototype, self.model)
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

        print("[INFO] computing object detections...")
        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > self.confidence:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                label = "{}: {:.2f}%".format(self.CLASSES[idx], confidence * 100)
                print("[INFO] {}".format(label))
                cv2.rectangle(image, (startX, startY), (endX, endY),
                              self.COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(image, label, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS[idx], 2)

        _, buffer = cv2.imencode('.jpg', image)

        return base64 \
            .b64encode(buffer) \
            .decode('utf-8')

        # cv2.imshow("Output", image)
        # cv2.waitKey(0)
