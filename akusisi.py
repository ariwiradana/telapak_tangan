import cv2
import numpy as np
from preprocessing import brightness_contrast


def roi(image):
    img_copy = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

    cont, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    M = cv2.moments(thresh)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # circle = cv2.circle(img_copy, (cX, cY), 7, (255, 255, 255), -1)

    x = int(cX - (cX * .2))
    y = int(cY - (cY * .3))
    w = int(cX + (cX * .6))
    h = int(cY + (cY * .6))

    cropped = img_copy[y:h, x:w]

    return cropped


def resize(img):
    shape = (200, 200)
    resize_img = cv2.resize(img, shape)
    return resize_img

