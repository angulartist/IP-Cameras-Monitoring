import os
import shutil

DATASET_PATH = './weapons_dataset/'
TRAIN_PATH = './training/images/train/weapon/'
TEST_PATH = './training/images/test/weapon/'

arr = os.listdir(TRAIN_PATH)

for item in arr:
    name, _ = item.split('.')
    related_image = f'{DATASET_PATH}{name}.jpg'
    shutil.copy(related_image, TRAIN_PATH)
