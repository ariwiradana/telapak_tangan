import cv2
import numpy as np


def grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def blur(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    return blur


def brightness_contrast(img):
    brightness = 10
    contrast = 300
    img = np.int16(img)
    img = img * (contrast / 127 + 1) - contrast + brightness
    img = np.clip(img, 0, 255)
    img = np.uint8(img)

    return img


def gabor_filter(img):
    kernel_size = 30
    sigma = 1
    theta = 1 * np.pi / 4
    lamda = 1 * np.pi / 4
    gamma = 0.02
    phi = 0
    kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma, theta, lamda, gamma, phi)
    return kernel


def canny_edge(img):
    edges = cv2.Canny(img, 20, 200)
    return edges


def threshold(img):
    threshold_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 15)
    return threshold_img
