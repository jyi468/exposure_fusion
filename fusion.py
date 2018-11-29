# from __future__ import division
import numpy as np
import cv2
import math
import scipy as sp
from matplotlib import pyplot as plt


def get_weights(img):
    print ''


def contrast(img, w_c=1):
    """Contrast is determined by absolute value of laplacian"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.absolute(cv2.Laplacian(gray, cv2.CV_8U)) ** w_c


def saturation(img, w_s=1):
    """Saturation is standard deviation within the RGB Channels"""
    return np.std(img, axis=2) ** w_s


def exposedness(img, w_e=1, sigma=0.2):
    """Well-exposedness is raw pixel intensities of a channel. We scale from 0 to 1
    we also apply this to each channel separately"""

    # Scale channels to be between 0 and 1. Convert to floating point numbers
    img = img / 255.0

    channels = np.empty([img.shape[0], img.shape[1], 3])
    for c in range(img.shape[2]):
        channel = img[:, :, c]
        channels[:, :, c] = np.exp((-channel - 0.5) ** 2 / (2 * sigma ** 2))

    return channels[:, :, 0] * channels[:, :, 1] * channels[:, :, 2]


def create_pyramid(img):
    print ''


def exposure_fusion(img_stack):
    weights = np.empty([len(img_stack), img_stack[0].shape[0], img_stack[0].shape[1]]).astype(np.uint8)
    # weights = []
    for i, img in enumerate(img_stack):
        c = contrast(img)
        s = saturation(img)
        e = exposedness(img)
        weights[i, :, :] = c * s * e
        # weights.append(c * s * e)

    # Normalize values
    weights = weights / sum(weights)

    img_stack = np.array(img_stack)
    r = np.empty([img_stack[0].shape[0], img_stack[0].shape[1], img_stack.shape[3]])
    for i in range(img_stack.shape[3]):
        r[:, :, i] = np.sum(img_stack[:, :, :, i] * weights, axis=0)

    print 'done'
    return r.astype(np.uint8)
