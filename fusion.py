# from __future__ import division
import numpy as np
import cv2
import math
import scipy as sp
from matplotlib import pyplot as plt


def get_weights(img_stack):
    weights = []
    for i, img in enumerate(img_stack):
        # Convert image to have intensities between 0 and 1
        img = img.astype(np.float32) / 255
        c = contrast(img)
        s = saturation(img)
        e = exposedness(img)
        weights.append(c * s * e)

    # Normalize values
    sum_w = np.sum(weights, axis=0)
    weights = np.divide(weights, sum_w, where=sum_w != 0)

    return weights


def contrast(img, w_c=1):
    """Contrast is determined by absolute value of laplacian"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.absolute(cv2.Laplacian(gray, cv2.CV_32F)) ** w_c


def saturation(img, w_s=1):
    """Saturation is standard deviation within the RGB Channels"""
    return np.std(img, axis=2, dtype=np.float32) ** w_s


def exposedness(img, w_e=1, sigma=0.2):
    """Well-exposedness is raw pixel intensities of a channel."""
    # Calculate values for each element and then multiply across channels
    return np.prod(np.exp(-((img - 0.5) ** 2 / (2 * sigma ** 2))), axis=2, dtype=np.float32) ** w_e


def gaussian_pyramid(img, depth):
    down = img.copy()
    pyr = [down]
    for i in range(depth):
        down = cv2.pyrDown(down).astype(np.float32)
        pyr.append(down)

    return pyr


def laplacian_pyramid(gPyr):
    lPyr = []
    for i in range(len(gPyr) - 1):
        expanded = cv2.pyrUp(gPyr[i + 1]).astype(np.float32)
        if expanded.shape[0] > gPyr[i].shape[0]:
            expanded = np.delete(expanded, 0, axis=0)
        if expanded.shape[1] > gPyr[i].shape[1]:
            expanded = np.delete(expanded, 0, axis=1)
        laplacian = cv2.subtract(gPyr[i], expanded)
        lPyr.append(laplacian)

    lPyr.append(gPyr[-1])
    return lPyr


def create_final_laplacian(gps, lps, depth, img_stack):
    # Multiply and Sum for output Laplacian
    lpR = []
    for l in range(depth):
        summed = None
        # for each image
        for k in range(img_stack.shape[0]):
            gaussian = gps[k][l]
            laplacian = lps[k][l]
            channel = np.empty_like(laplacian)
            # for each color channel
            for c in range(3):
                channel[:, :, c] = gaussian * laplacian[:, :, c]

            summed = channel

        lpR.append(summed)

    return lpR


def collapse(lPyr):
    collapsed = lPyr[len(lPyr) - 1]

    for i in range(len(lPyr) - 1, 0, -1):
        expanded = cv2.pyrUp(collapsed).astype(np.float32)

        if expanded.shape[0] > lPyr[i - 1].shape[0]:
            expanded = np.delete(expanded, 0, axis=0)
        if expanded.shape[1] > lPyr[i - 1].shape[1]:
            expanded = np.delete(expanded, 0, axis=1)

        collapsed = expanded + lPyr[i - 1]

    return collapsed


def exposure_fusion(img_stack):
    img_stack = np.array(img_stack).astype(np.float32)
    weights = get_weights(img_stack)
    depth = int(np.log2(min(img_stack[0].shape[:2])))

    gps = []
    # Gaussian of weights
    for i in range(weights.shape[0]):
        gpW = gaussian_pyramid(weights[i], depth)
        gps.append(gpW)

    lps = []
    # Get Laplacian pyramid of for each image
    for i in range(img_stack.shape[0]):
        gpI = gaussian_pyramid(img_stack[i, :, :, :], depth)
        lpI = laplacian_pyramid(gpI)
        lps.append(lpI)

    lpR = create_final_laplacian(gps, lps, depth, img_stack)
    # Collapse pyramid
    output = collapse(lpR)
    return output.astype(np.uint8)
