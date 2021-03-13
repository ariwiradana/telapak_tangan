import cv2
import numpy as np
from matplotlib import pyplot as plt

from akusisi import crop, roi, resize
from preprocessing import grayscale, canny_edge, blur, brightness_contrast, threshold


def gabor_filter(img):
    ksize = 40
    sigma = 1*np.pi/4
    theta = 1*np.pi/4
    lamda = 10
    gamma = 1
    phi = 0

    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
    filtered_img = cv2.filter2D(img, -1, kernel)
    return filtered_img


img = cv2.imread('Dataset/Palmprint/img0001_m_r_02.jpg')
img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = crop(img)
img = roi(img)
img = resize(img)


# img = gabor_filter(img)
plt.imshow(img, cmap='gray')
plt.show()
