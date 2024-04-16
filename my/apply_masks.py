import os
import random
from typing import Tuple, List

import cv2
import numpy as np
import math


if __name__ == "__main__":
    base_dir = "/home/m/datasets/Novi11Cropped"
    masks = [f for f in os.listdir(base_dir) if "mask" in f]
    for mask in masks:
        mask = os.path.join(base_dir, mask)
        mask_img = cv2.imread(mask)
        actual_img = cv2.imread(mask.replace("mask",""))
        actual_img[mask_img == 0] = 0
        cv2.imwrite(mask.replace("mask", "").replace(".png", "Masked.png"), actual_img)