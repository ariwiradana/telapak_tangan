import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops
from akusisi import roi, resize
from preprocessing import grayscale, sharpen, brightness_contrast, blur

filename = "Dataset/Palmprint/img0134_m_r_02.jpg"

img_ori = cv2.imread(filename)
rotate = cv2.rotate(img_ori, cv2.ROTATE_90_CLOCKWISE)
roi = roi(rotate)
resize = resize(roi)

gray = grayscale(resize)
sharpen = sharpen(gray)
kontras = brightness_contrast(sharpen)
smooth = blur(kontras)


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