import cv2
import matplotlib.pyplot as plt
import numpy as np
from akusisi import roi, resize
from preprocessing import grayscale, brightness_contrast, blur, sharpen, edges, non_max_suppression


def show(img):
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap="gray")
    plt.show()
    return img


# akuisisi
filename = "Dataset/Palmprint/img0108_m_l_08.jpg"

img_ori = cv2.imread(filename)
rotate = cv2.rotate(img_ori, cv2.ROTATE_90_CLOCKWISE)
roi = roi(rotate)
resize = resize(roi)

# preprocessing
gray = grayscale(resize)
sharpen = sharpen(gray)
kontras = brightness_contrast(sharpen)
smooth = blur(kontras)
edge = edges(smooth)

plt.figure(figsize=(7, 5))
plt.subplot(2, 3, 1), plt.imshow(gray, cmap="gray"), plt.xticks([]), plt.yticks([]), plt.xlabel("Grayscale")
plt.subplot(2, 3, 2), plt.imshow(sharpen, cmap="gray"), plt.xticks([]), plt.yticks([]), plt.xlabel("Sharpen")
plt.subplot(2, 3, 3), plt.imshow(kontras, cmap="gray"), plt.xticks([]), plt.yticks([]), plt.xlabel("Kontras")
plt.subplot(2, 3, 4), plt.imshow(smooth, cmap="gray"), plt.xticks([]), plt.yticks([]), plt.xlabel("Smooth")
plt.subplot(2, 3, 5), plt.imshow(edge, cmap="gray"), plt.xticks([]), plt.yticks([]), plt.xlabel("Edges")
plt.show()
