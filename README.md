# mask_utils

Purpose: This is a helper library with common functions I need on a daily basis while dealing with semantic segmentation data.

Note: This repo is under active development and things might break, use with caution until a tested release is created.

## Proposed functions to build
* check if image read is binary
* scale pixel intensity automatically to 0-255 range
* pixel count of foreground vs background
* percentage area of image occupied by foreground
* if mask is empty or not
* if mask is single channeled or not
* compare and choose channel(with most information) if multi-channel image is used as input
* threshold mask for given thresholds
* invert binary mask
* convert mask image to uint8
* count instances of foreground present
* save mask as png image
* convert mask to coco annotation format
* for softmax, combine multiple masks into single mask with intensity mapping
* for softmax, explode single mask with intensity mapping into respective separate masks
* display a coloured mask for viewing multi intensity masks
* erosion and dilation of mask data
* mask to rle
* rle to mask
* coco format to mask image
* feather a mask
* draw square around mask
* find maximum vertical and horizontal expanse of mask(in pixels)
* convert mask to contour boundary
* find centroid of foreground (if multiple shapes are present, find centroid of all shapes)
* numpy to pillow
* pillow to numpy
* save mask to destination
* semantic segmentation metrics
    * dice score
    * iou
    * precision
    * recall
    * accuracy
    * mAP