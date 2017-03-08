import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2


def transform(img):
    h, w = img.shape[0], img.shape[1]

    src = np.float32(
        [[100, h],
        [(w / 2) - 90, h / 2 + 110],
        [(w / 2 + 90), h / 2 + 110],
        [w-100, h]])

    dst = np.float32(
        [[100, h],
        [100, 0],
        [w - 100, 0],
        [w - 100, h]])


    M = cv2.getPerspectiveTransform(src, dst)
    MI = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)

    return warped, MI
