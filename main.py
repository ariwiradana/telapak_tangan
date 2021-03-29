import os

import cv2
import pandas as pd
from tqdm import tqdm

from akusisi import resize, roi
from preprocessing import grayscale, brightness_contrast
from test import calc_glcm_all_agls


def load_image(IMG_DIR):
    images = []
    labels = []

    for filename in tqdm(os.listdir(IMG_DIR)):
        if filename.split('.')[1] == 'jpg':
            # akuisisi
            img = cv2.imread(os.path.join(IMG_DIR, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img = roi(img)
            img = resize(img)

            # preprocessing
            img = grayscale(img)
            img = brightness_contrast(img)

            images.append(img)
            labels.append(filename.split('_')[0])
    return images, labels


def glcm(img_dir):
    images, labels = load_image(img_dir)
    properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']

    glcm_all_agls = []
    for img, label in zip(images, labels):
        glcm_all_agls.append(calc_glcm_all_agls(img, label, props=properties))

    columns = []
    angles = ['0', '45', '90', '135']
    for name in properties:
        for ang in angles:
            columns.append(name + "_" + ang)

    columns.append("label")

    glcm_df = pd.DataFrame(glcm_all_agls, columns=columns)
    # glcm_df.to_csv("fitur.csv")

    return glcm_df


IMG_DIR = 'Dataset/Palmprint'
glcm = glcm(IMG_DIR)
print(glcm)
