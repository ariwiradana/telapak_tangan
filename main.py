import os

import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from akusisi import resize, crop, roi
from preprocessing import grayscale
from training import encode, split, cnn, training


def load_image(IMG_DIR):
    list_img = []
    list_label = []

    for filename in tqdm(os.listdir(IMG_DIR)):
        if filename.split('.')[1] == 'jpg':
            img = cv2.imread(os.path.join(IMG_DIR, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img = crop(img)
            img = roi(img)
            img = resize(img)

            list_img.append(img)
            list_label.append(filename.split('_')[0])
    return list_img, list_label


IMG_DIR = 'Dataset/Palmprint'
X, y = load_image(IMG_DIR)

print(y)

