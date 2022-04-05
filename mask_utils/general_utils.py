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

def is_single_channeled(mask: np.ndarray) -> bool:
    channels = get_channels(mask)
    return bool(channels == 1)


def is_empty(mask: np.ndarray) -> bool:
    unique_pixel_intensities = np.unique(mask)
    if len(unique_pixel_intensities) == 1:
        if unique_pixel_intensities[0] == 0:
            return True
    elif len(unique_pixel_intensities) > 2:
        return False


def to_uint8(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask, dtype=np.uint8)


def pil_to_np(mask: Image) -> np.ndarray:
    if isinstance(mask, np.ndarray):
        return mask
    elif isinstance(mask, Image):
        return np.asarray(mask)
    else:
        print('Invalid format')

def np_to_pil(mask: np.ndarray) -> Image:
    if isinstance(mask, np.ndarray):
        return Image.fromarray(mask)
    elif isinstance(mask, Image):
        return mask
    else:
        print('Invalid format')


def mask_to_rle(mask):
    # credits: https://www.kaggle.com/code/kambarakun/mask-images-rle-strings-speed-test-4-methods/notebook
    # method 1: https://www.kaggle.com/stainsby/fast-tested-rle
    if np.max(mask) == 255:
        mask = mask/255
    assert is_empty(mask) or np.max(mask) == 1
    pixels = mask.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return ' '.join(str(x) for x in runs)


def rle_to_mask(rle_string, height, width, max_intensity=255):
    rows, cols = height, width
    rle_numbers = [int(num_string) for num_string in rle_string.split(' ')]
    rle_pairs = np.array(rle_numbers).reshape(-1,2)
    mask = np.zeros(rows*cols,dtype=np.uint8)
    for index, length in rle_pairs:
        index -= 1
        mask[index:index+length] = max_intensity
    mask = mask.reshape(cols,rows)
    mask = mask.T
    return mask

def mask_to_base64():
    pass

def base64_to_mask():
    pass

def feather_mask():
    pass

def detect_instances():
    pass

def max_stretches():
    pass


def mask_to_contours():
    pass


def find_centroids():
    pass

def is_binary():
    pass


def scale_uint8():
    pass


def pixel_count():
    pass

def percentage_fg():
    pass


def invert():
    pass


def get_instance_count():
    instance_count = detect_instances()

def mask_to_coco():
    pass


def coco_to_mask():
    pass


def make_3_channeled():
    pass


def stack_vertical():
    pass


def stack_horizontal():
    pass


def mask_to_pascal_voc():
    pass


def pascal_voc_to_mask():
    pass


def dice_score():
    pass


def iou_score():
    pass

def precision_score():
    pass


def recall_score():
    pass

def accuracy():
    pass


def balanced_accuracy():
    pass

def mAP_score():
    pass