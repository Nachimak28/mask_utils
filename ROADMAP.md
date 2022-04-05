## Proposed functions to build

Note: No feature has any priority/order
* General functions
    * if mask is single channeled or not
    * if mask is empty or not
    * convert mask image to uint8
    * save mask as png image
    * mask to rle
    * rle to mask
    * mask to b64
    * b64 to mask
    * feather a mask
    * numpy to pillow
    * pillow to numpy
    * draw square around mask
    * find maximum vertical and horizontal expanse of mask(in pixels)
    * convert mask to contour boundary
    * find centroid of foreground (if multiple shapes are present, find centroid of all shapes)
* Binary masks
    * check if image read is binary
    * scale pixel intensity automatically to 0-255 range
    * pixel count of foreground vs background
    * percentage area of image occupied by foreground
    * compare and choose channel(with most information) if multi-channel image is used as input
    * threshold mask for given thresholds
    * invert binary mask
    * count instances of foreground present
    * convert mask to coco annotation format
    * coco format to mask image
    * make 3 channeled
    * stack multiple masks together on top of each other
    * stack multiple masks side by side
    * erosion, dilation, opening, closing of mask data
* Categorical masks
    * for categorical masks, combine multiple masks into single mask with intensity mapping
    * for categorical masks, explode single mask with intensity mapping into respective separate masks
    * for categorical masks, identify overlap of multiple masks
    * display a coloured mask for viewing multi intensity masks


* mask to pascal voc format
* pascal voc format to mask image
* save mask to destination
* semantic segmentation metrics
    * dice score
    * iou
    * precision
    * recall
    * accuracy
    * mAP
* build patches from mask image
* voxel functions (probably)