import os
import xml.etree.ElementTree as ET

import tensorflow as tf
from object_detection.utils import dataset_util

ROOT_FOLDER = 'training'
PATH_TEST = f"{ROOT_FOLDER}/images/eval/weapon/"
PATH_RECORD_TEST = f"{ROOT_FOLDER}/eval.record"
PATH_TRAIN = f"{ROOT_FOLDER}/images/train/weapon/"
PATH_RECORD_TRAIN = f"{ROOT_FOLDER}/train.record"

IMAGE_FORMAT = b'jpg'


def class_text_to_int(row_label):
    if row_label == 'weapon':
        return 1
    else:
        return None


def xml_to_tf(path_input, path_output):
    writer = tf.io.TFRecordWriter(path_output)

    files = os.listdir(path_input)
    for file in files:
        if file.endswith(".xml"):
            xmlFile = path_input + file

            tree = ET.parse(xmlFile)
            root = tree.getroot()

            filename = root.find('filename').text
            width = int(root.find('size')[0].text)
            height = int(root.find('size')[1].text)

            x_mins = []
            x_maxs = []
            y_mins = []
            y_maxs = []
            classes_text = []
            classes = []

            for member in root.findall('object'):
                name = member[0].text
                x_min = int(member[5][0].text)
                y_min = int(member[5][1].text)
                x_max = int(member[5][2].text)
                y_max = int(member[5][3].text)

                x_mins.append(x_min / width)
                x_maxs.append(x_max / width)
                y_mins.append(y_min / height)
                y_maxs.append(y_max / height)
                classes_text.append(name.encode('utf8'))
                classes.append(class_text_to_int(name))

            with tf.io.gfile.GFile(os.path.join(path_input, '{}'.format(filename)), 'rb') as fid:
                encoded_jpg = fid.read()
            tf_example = tf.train.Example(features=tf.train.Features(feature={
                    'image/height'            : dataset_util.int64_feature(height),
                    'image/width'             : dataset_util.int64_feature(width),
                    'image/filename'          : dataset_util.bytes_feature(filename.encode('utf8')),
                    'image/source_id'         : dataset_util.bytes_feature(filename.encode('utf8')),
                    'image/encoded'           : dataset_util.bytes_feature(encoded_jpg),
                    'image/format'            : dataset_util.bytes_feature(IMAGE_FORMAT),
                    'image/object/bbox/xmin'  : dataset_util.float_list_feature(x_mins),
                    'image/object/bbox/xmax'  : dataset_util.float_list_feature(x_maxs),
                    'image/object/bbox/ymin'  : dataset_util.float_list_feature(y_mins),
                    'image/object/bbox/ymax'  : dataset_util.float_list_feature(y_maxs),
                    'image/object/class/text' : dataset_util.bytes_list_feature(classes_text),
                    'image/object/class/label': dataset_util.int64_list_feature(classes),
            }))

            writer.write(tf_example.SerializeToString())
    writer.close()
    output_path = os.path.join(os.getcwd(), path_output)
    print('Successfully created the TFRecords: {}'.format(output_path))


xml_to_tf(PATH_TEST, PATH_RECORD_TEST)
xml_to_tf(PATH_TRAIN, PATH_RECORD_TRAIN)
