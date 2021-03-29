import os

import cv2
from skimage.feature import greycomatrix, greycoprops
import numpy as np
from tqdm import tqdm
import pandas as pd

from akusisi import resize, roi
from preprocessing import grayscale, brightness_contrast, sharpen, blur
import matplotlib.pyplot as plt

# filename = "Dataset/Palmprint/img0134_m_r_02.jpg"
#
# img_ori = cv2.imread(filename)
# rotate = cv2.rotate(img_ori, cv2.ROTATE_90_CLOCKWISE)
# roi = roi(rotate)
# resize = resize(roi)
#
# sharpen = sharpen(resize)
# kontras = brightness_contrast(sharpen)
# smooth = blur(kontras)
# gray = grayscale(smooth)
#
# labels = ['test gambar']


def calc_glcm_all_agls(img, label, props, dists=[5], agls=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], lvl=256, sym=True,
                       norm=True):
    glcm = greycomatrix(img,
                        distances=dists,
                        angles=agls,
                        levels=lvl,
                        symmetric=sym,
                        normed=norm)
    feature = []
    glcm_props = [propery for name in props for propery in greycoprops(glcm, name)[0]]
    for item in glcm_props:
        feature.append(item)
    feature.append(label)

    return feature

