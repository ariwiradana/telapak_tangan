import cv2
import numpy as np
from scipy import ndimage


def grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def brightness_contrast(img):
    brightness = -30
    contrast = 110
    img = np.int16(img)
    img = img * (contrast / 127 + 1) - contrast + brightness
    img = np.clip(img, 0, 255)
    img = np.uint8(img)
    return img


def blur(img):
    blurred = cv2.GaussianBlur(img, (3, 3), 5)
    return blurred


def gabor_filter(img):
    kernel_size = 30
    sigma = 1
    theta = 1 * np.pi / 4
    lamda = 1 * np.pi / 4
    gamma = 0.02
    phi = 0
    kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma, theta, lamda, gamma, phi)
    gabor = cv2.filter2D(img, -1, kernel)
    return gabor


def sharpen(img):
    filter = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    # Applying cv2.filter2D function on our Cybertruck image
    sharpen = cv2.filter2D(img, -1, filter)
    return sharpen


def edges(img):
    low_thresh = 110
    high_tresh = 120
    edges = cv2.Canny(img, low_thresh, high_tresh)
    return edges


def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255

                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                # angle 45
                elif 22.5 <= angle[i, j] < 67.5:
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                # angle 90
                elif 67.5 <= angle[i, j] < 112.5:
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # angle 135
                elif 112.5 <= angle[i, j] < 157.5:
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass

    return Z