import cv2 as cv
import tensorflow as tf

FROZEN_INFERENCE_GRAPH_PATH = './ml_processing/model/trained/frozen_inference_graph.pb'


class Detector(object):
    def __init__(self):
        with tf.gfile.FastGFile(FROZEN_INFERENCE_GRAPH_PATH, 'rb') as f:
            self.graph_def = tf.compat.v1.GraphDef()
            self.graph_def.ParseFromString(f.read())

        self.sess = tf.compat.v1.Session()
        self.sess.graph.as_default()
        tf.import_graph_def(self.graph_def, name='')

    def detect(self, img):
        rows = img.shape[0]
        cols = img.shape[1]
        inp = cv.resize(img, (300, 270))
        inp = inp[:, :, [2, 1, 0]]

        # Run the model
        out = self.sess.run([
            self.sess.graph.get_tensor_by_name('num_detections:0'),
            self.sess.graph.get_tensor_by_name('detection_scores:0'),
            self.sess.graph.get_tensor_by_name('detection_boxes:0'),
            self.sess.graph.get_tensor_by_name('detection_classes:0')
        ], feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

        # Visualize detected bounding boxes.
        num_detections = int(out[0][0])
        for i in range(num_detections):
            _ = int(out[3][0][i])  # Index
            score = float(out[1][0][i])
            bbox = [float(v) for v in out[2][0][i]]

            if score > 0.3:
                x = bbox[1] * cols
                y = bbox[0] * rows
                right = bbox[3] * cols
                bottom = bbox[2] * rows
                cv.rectangle(img,
                             (int(x), int(y)),
                             (int(right), int(bottom)),
                             (125, 255, 51),
                             thickness=2)

        return img
