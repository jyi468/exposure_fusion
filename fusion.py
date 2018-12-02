import numpy as np
import cv2


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

    return (weights * 255).astype(np.uint8)


def contrast(img, w_c=1):
    """Contrast is determined by absolute value of laplacian"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.absolute(cv2.Laplacian(gray, cv2.CV_32F)) ** w_c + 1


def saturation(img, w_s=1):
    """Saturation is standard deviation within the RGB Channels"""
    return np.std(img, axis=2, dtype=np.float32) ** w_s + 1


def exposedness(img, w_e=1, sigma=0.2):
    """Well-exposedness is raw pixel intensities of a channel."""
    # Calculate values for each element and then multiply across channels
    return np.prod(np.exp(-((img - 0.5) ** 2 / (2 * sigma ** 2))), axis=2, dtype=np.float32) ** w_e + 1


def gaussian_kernel(size=5, sigma=0.4):
    return cv2.getGaussianKernel(ksize=size, sigma=sigma)


def image_reduce(image):
    kernel = gaussian_kernel()
    if len(image.shape) == 3:
        out_image = cv2.filter2D(image, cv2.CV_8UC3, kernel)
    else:
        out_image = cv2.filter2D(image, cv2.CV_8U, kernel)
    out_image = cv2.resize(out_image, None, fx=0.5, fy=0.5)
    return out_image


def image_expand(image):
    kernel = gaussian_kernel()
    out_image = cv2.resize(image, None, fx=2, fy=2)
    out_image = cv2.filter2D(out_image, cv2.CV_8UC3, kernel)
    return out_image


def gaussian_pyramid(img, depth):
    reduced = img.copy()
    pyr = [reduced]
    for i in range(depth):
        reduced = image_reduce(reduced)
        pyr.append(reduced)

    return pyr


def laplacian_pyramid(gPyr):
    lPyr = [gPyr[-3]]
    for i in range(len(gPyr) - 3, 0, -1):
        expanded = image_expand(gPyr[i])
        if expanded.shape[0] > gPyr[i - 1].shape[0]:
            expanded = np.delete(expanded, 0, axis=0)
        if expanded.shape[1] > gPyr[i - 1].shape[1]:
            expanded = np.delete(expanded, 0, axis=1)
        laplacian = cv2.subtract(gPyr[i - 1], expanded)
        lPyr = [laplacian] + lPyr

    return lPyr


def create_final_laplacian(gPyrs, lPyrs, depth, img_stack):
    # Multiply and Sum for output Laplacian
    lpR = []
    # for each gaussian/laplacian
    for l in range(depth):
        summed = np.zeros(lPyrs[0][l].shape, dtype=np.uint8)
        # for each image
        for k in range(img_stack.shape[0]):
            gaussian = gPyrs[k][l].astype(np.float32) / 255
            laplacian = lPyrs[k][l]
            channel = np.empty_like(laplacian)
            # for each color channel
            for c in range(3):
                channel[:, :, c] = gaussian * laplacian[:, :, c]
            gaussian = np.dstack((gaussian, gaussian, gaussian))
            summed += cv2.multiply(gaussian, laplacian, dtype=cv2.CV_8UC3)

            # summed = channel

        lpR.append(summed)

    return lpR


def collapse(lPyr):
    collapsed = lPyr[len(lPyr) - 1]

    for i in range(len(lPyr) - 1, 0, -1):
        expanded = image_expand(collapsed)

        if expanded.shape[0] > lPyr[i - 1].shape[0]:
            expanded = np.delete(expanded, 0, axis=0)
        if expanded.shape[1] > lPyr[i - 1].shape[1]:
            expanded = np.delete(expanded, 0, axis=1)

        collapsed = expanded + lPyr[i - 1]

    return collapsed


def exposure_fusion(img_stack):
    img_stack = np.array(img_stack)
    # weights = compute_weights(img_stack, None)
    weights = get_weights(img_stack)
    # depth = int(np.log2(min(img_stack[0].shape[:2]))) - 5
    depth = 4
    # # Weights only
    # r = np.empty([img_stack[0].shape[0], img_stack[0].shape[1], img_stack.shape[3]])
    # # For each channel
    # for i in range(img_stack.shape[3]):
    #     r[:, :, i] = np.sum(weights * img_stack[:, :, :, i], axis=0)

    # return r

    # All
    gPyrs = []
    # Gaussian of weights
    for i in range(weights.shape[0]):
        gpW = gaussian_pyramid(weights[i], depth)
        gPyrs.append(gpW)

    lPyrs = []
    # Get Laplacian pyramid of for each image
    for i in range(img_stack.shape[0]):
        gpI = gaussian_pyramid(img_stack[i, :, :, :], depth + 1)
        lpI = laplacian_pyramid(gpI)
        lPyrs.append(lpI)

    lpR = create_final_laplacian(gPyrs, lPyrs, depth, img_stack)

    # Collapse pyramid
    output = collapse(lpR)
    return output
