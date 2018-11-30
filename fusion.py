# from __future__ import division
import numpy as np
import cv2
import math
import scipy as sp
from matplotlib import pyplot as plt


def get_weights(img_stack):
    # weights = np.empty([len(img_stack), img_stack[0].shape[0], img_stack[0].shape[1]]).astype(np.float32)
    weights = []
    for i, img in enumerate(img_stack):
        # Convert image to have intensities between 0 and 1
        img = np.float32(img) / 255
        c = contrast(img, i)
        s = saturation(img)
        e = exposedness(img)
        # weights[i, :, :] = c * s * e
        weights.append(c * s * e)

    # Normalize values
    sum_w = np.sum(weights, axis=0)
    weights = np.divide(weights, sum_w, where=sum_w != 0)

    return weights


def contrast(img, i, w_c=1):
    """Contrast is determined by absolute value of laplacian"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.absolute(cv2.Laplacian(gray, cv2.CV_32F)) ** w_c


def saturation(img, w_s=1):
    """Saturation is standard deviation within the RGB Channels"""
    return np.std(img, axis=2, dtype=np.float32) ** w_s


def exposedness(img, w_e=1, sigma=0.2):
    """Well-exposedness is raw pixel intensities of a channel. We scale from 0 to 1
    we also apply this to each channel separately"""

    # Calculate values for each element and then multiply across channels
    return np.prod(np.exp(-((img - 0.5) ** 2 / (2 * sigma ** 2))), axis=2, dtype=np.float32) ** w_e


def create_pyramid(img):
    print ''


def exposure_fusion(img_stack):
    weights = get_weights(img_stack)
    img_stack = np.array(img_stack)
    r = np.empty([img_stack[0].shape[0], img_stack[0].shape[1], img_stack.shape[3]], dtype=np.uint8)
    for i in range(img_stack.shape[3]):
        r[:, :, i] = np.sum(img_stack[:, :, :, i] * weights, axis=0)

    return np.uint8(r)
