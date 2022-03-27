import os
import numpy as np
import imageio as io
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import config

class Mask:
    def __init__(self, mask_path, default_extension=config.DEFAULT_EXTENSION):
        self.mask_path = mask_path
        self.default_extension = default_extension
        self.mask_image_name = os.path.splitext(os.path.split(self.mask_path)[-1])[0]
        self.data = io.imread(self.mask_path)
        self.data_type = np.uint8
    
    def to_pil(self):
        return Image.fromarray(self.data)
    
    def _determine_channels(self):
        pass

    def save(self, destination_path, new_name=None):
        if os.path.exists(destination_path):
            if new_name is not None:
                # check if with or without extension
                output_image_name = new_name
            io.imwrite(os.path.join(destination_path, output_image_name), self.data)
        else:
            raise FileNotFoundError
