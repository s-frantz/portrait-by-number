"""
Class for getting masks from raw image.
"""

import cv2
import numpy as np
from utils.ai import face_seg
import utils.image as im_utils


PBN_CLASSES = {
    1: [ # FACE
        1, # skin of face
        2, # left eyebrow
        3, # right eyebrow
        4, # left eye
        5, # right eye
        7, # left ear
        8, # right ear
        10, # nose
        11, # mouth
        12, # upper lip
        13, # lower lip
    ],
    2: [ # FRAMING
        6, # eye glasses
        9, # ear ring
        14, # skin of neck
        15, # necklace
        17, # hair
        18, # hat
    ],
    3: [ # CLOTHING
        16, #clothing
    ],
    4: [ # BACKGROUND
        0, #no face seg class
    ],
}


def resize_to_image(im, to_im):
    s = to_im.shape
    return cv2.resize(im, (s[0], s[1]))


def map_to_pbn(im_seg):
    """
    Converts face seg classes to paint-by-number classes
    """
    pbn_map = {
        v_: k
        for k, v in PBN_CLASSES.items()
        for v_ in v
    }
    mp = np.vectorize(lambda x: pbn_map[x])
    return mp(im_seg)


def remove_slivers(m, buffer=8):
    """
    Remove slivers from mask using neg buff + pos buff combo.
    """
    inv = 1.0 * im_utils.invert(m)
    invbuff = im_utils.buffer_mask(inv, buffer=buffer)
    mneg = 1.0 * im_utils.invert(invbuff)
    return im_utils.buffer_mask(mneg, buffer=buffer)


class Masks:
    
    def __init__(self, im: np.ndarray, buffer: int = 7):
        
        """
        Inferences the input image using the face segmentation model.
        Converts the resulting mask to the four paint-by-number classes:
            1. face
            2. face framing
            3. clothes
            4. background
        Finally, removes slivers and returns a list of binary masks, one for each category.
        
        Args:
            im (numpy.ndarray): RGB image
            buffer (int): number of pixels to buffer masks by, default 6
        """
        # params to class vars
        self.im = im
        self.buffer = buffer
        
        # run image segmentation
        _, seg_mask = face_seg.evaluate(self.im)
        
        # resize to original
        seg_mask_orig_size = resize_to_image(seg_mask, self.im)
        
        # map to our four pbn classes
        self.seg_mask = map_to_pbn(seg_mask_orig_size)
        
        # fill a dict of binary classes
        self.binary_masks = [
            self.get_binary(x + 1)
            for x in range(4)
        ]
        
        ## get combined raster
        self.seg_mask = self.get_combined()
        
    def get_binary(self, class_int):
        """
        Extract binary mask, remove slivers, and buffer.
        """
        # create np array of zeros of orig image size
        z = np.zeros(self.im.shape[:2])
        
        # fill in as 1's for specific mask value
        z[self.seg_mask==class_int] = 1.0
        
        # background has no slivers, otherwise remove them
        if class_int != 4:
            z = remove_slivers(z)
        
        # buffer
        return im_utils.buffer_mask(z, self.buffer).astype(np.float32)

    def get_combined(self):
        """
        Iterate through masks and combine back to a single image
        """
        # create np array of zeros of orig image size
        z = np.zeros(self.im.shape[:2])
        
        for ix in range(len(self.binary_masks), 0, -1):
            z[self.binary_masks[ix-1]==1] = int(ix)
        
        return z.astype(np.uint8)
