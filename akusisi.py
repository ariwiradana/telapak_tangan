import cv2
import numpy as np
from preprocessing import brightness_contrast


def roi(img):
    height, width = img.shape[:2]
    start_row, start_col = int(height * .4), int(width * .1)
    end_row, end_col = int(height * .9), int(width * .7)
    cropped = img[start_row:end_row, start_col:end_col]
    return cropped


def crop(img):
    lower_range = np.array([90])
    higher_range = np.array([255])

    img_copy = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_kontras = brightness_contrast(img_gray)

    mask = cv2.inRange(img_kontras, lower_range, higher_range)

    cont, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    kontur = cv2.drawContours(img, cont, -1, 255, 3)
    c = max(cont, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(kontur, (x, y), (x + w, y + h), (0, 255, 0), 20)

    cropped = img_copy[y:y + h, x:x + w]

    return cropped


def resize(img):
    shape = (300, 300)
    resize_img = cv2.resize(img, shape)
    return resize_img

