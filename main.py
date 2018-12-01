import numpy as np
import cv2
import fusion as fusion
from os import path
import os
import errno

SRC_FOLDER = "images/source/marble2"
OUT_FOLDER = "images/output"
EXTENSIONS = set(["bmp", "jpeg", "jpg", "png", "tif", "tiff"])


def main(image_files, output_folder):
    img_stack = [cv2.imread(name) for name in image_files
                 if path.splitext(name)[-1][1:].lower() in EXTENSIONS]

    if any([im is None for im in img_stack]):
        raise RuntimeError("One or more input files failed to load.")

    fused_image = fusion.exposure_fusion(img_stack)
    cv2.imwrite(path.join(output_folder, "output.png"), fused_image)

    print "Done!"


if __name__ == "__main__":
    """Generate fusion images using input images"""

    np.random.seed()  # set a fixed seed if you want repeatable results

    src_contents = os.walk(SRC_FOLDER)
    dirpath, _, fnames = src_contents.next()

    image_dir = os.path.split(dirpath)[-1]
    output_dir = os.path.join(OUT_FOLDER, image_dir)

    try:
        os.makedirs(output_dir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    print "Processing '" + image_dir + "' folder..."

    image_files = sorted([os.path.join(dirpath, name) for name in fnames])

    main(image_files, output_dir)
