import os
import numpy as np
import imageio as io
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_channels(mask: np.ndarray) -> int:
    if mask.ndim == 2:
        return 1
    elif mask.ndim == 3:
        return mask.shape[2]
    else:
        raise Exception('Not a valid mask')


def is_empty(mask: np.ndarray) -> bool:
    unique_pixel_intensities = np.unique(mask)
    if len(unique_pixel_intensities) == 1:
        if unique_pixel_intensities[0] == 0:
            return True
    elif len(unique_pixel_intensities) > 2:
        return False


def to_uint8(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask, dtype=np.uint8)

def erosion()