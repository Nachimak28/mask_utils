import os
import cv2
import zlib
import base64
import io as bio
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

def mask_to_base64(mask):
    img_pil = Image.fromarray(np.array(mask, dtype=np.uint8))
    buffer = bio.BytesIO()
    img_pil.save(buffer, format='PNG', transparency=0, optimize=0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def base64_to_mask(bstring):
    decoded_data = base64.b64decode(bstring)
    pil_mask = Image.open(bio.BytesIO(decoded_data)).convert('L')
    mask = np.asarray(pil_mask, dtype=np.uint8)
    return mask


def dilate_mask(mask, kernel_size=11, iterations=5):
    kernel  = np.ones((kernel_size,kernel_size), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=iterations)
    return dilated

def feather_mask(mask, dilation_kernel_size=11, blur_kernel_size=99):
    dilated_mask = dilate_mask(mask=mask, kernel_size=dilation_kernel_size)
    feathered = cv2.GaussianBlur(dilated_mask,(blur_kernel_size,blur_kernel_size),0)
    return feathered

def bbox_full_mask(mask):
    x = (mask.sum(axis=0)!=0).argmax(axis=0)
    y = (mask.sum(axis=1)!=0).argmax(axis=0)
        
    height = (mask.sum(axis=1)!=0).sum()
    width = (mask.sum(axis=0)!=0).sum()

    return [x, y, width, height]


def detect_instances():
    pass

def max_stretches():
    pass


def mask_to_contours():
    pass


def find_centroids():
    pass

def is_binary(mask):
    unique_intensities = np.unique(mask)
    if len(unique_intensities) > 2:
        return False
    return True


def pixel_count(mask, pixel_intensity=255):
    return sum(mask == pixel_intensity)

def get_area(mask):
    return np.count_nonzero(mask)


def percentage_fg(mask, pixel_intensity=255):
    total_number_of_pixels = mask.flatten().shape[0]
    return (sum(mask==255)*100/total_number_of_pixels)


def invert(mask):
    max_intensity = np.max(mask)
    return max_intensity - mask


def get_instance_count():
    instance_count = detect_instances()

def mask_to_coco():
    pass


def coco_to_mask():
    pass


def make_3_channeled(mask):
    channels = get_channels(mask)
    three_channeled_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    if channels == 3:
        return mask
    elif channels < 3:
        if channels == 1:
            three_channeled_mask[:, :, 0] = mask
            three_channeled_mask[:, :, 1] = mask
            three_channeled_mask[:, :, 2] = mask
            return three_channeled_mask
        else:
            raise Exception('Invalid Mask')
    elif channels > 3:
        return mask[:, :, :3]



def stack_on_top(mask_list, channel_first=True):
    if channel_first:
        axis = -1
    else:
        axis = 0
    stacked_image = np.stack(mask_list, axis=axis)
    return stacked_image


def stack_sideways(mask_list):
    # ensure all masks in mask_list of same size and channels
    return np.hstack(mask_list)


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

def accuracy(mask_true, mask_pred):
    # TODO: Validate this with sklearn
    flat_mask_true = mask_true.flatten()
    flat_mask_pred = mask_pred.flatten()

    return np.sum(flat_mask_true == flat_mask_pred)/flat_mask_pred.shape[0]


def mAP_score():
    pass