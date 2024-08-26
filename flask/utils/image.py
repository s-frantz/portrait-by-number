import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(_i_path):
    return cv2.imread(_i_path)

def natural_color(_i):
    return cv2.cvtColor(_i, cv2.COLOR_BGR2RGB)

def lab_color(_i):
    return cv2.cvtColor(_i, cv2.COLOR_BGR2LAB)

def bgr_color(_i): # from rgb
    return cv2.cvtColor(_i, cv2.COLOR_RGB2BGR)

def display_image(_i):
    plt.imshow(_i)
    plt.colorbar()
    plt.show()
    
def save_image(_i, _i_path):
    cv2.imwrite(_i_path, _i)
    
def unique_colors(_im, return_counts=False):
    return np.unique(
        _im.reshape(-1, _im.shape[2]),
        axis=0,
        return_counts=return_counts
    )

def blur(_im, radius=5):
    return cv2.blur(_im, (radius, radius))

def invert(_im):
    return np.logical_not(_im)

def foreground_extractor(image, rectangle):
    #https://www.geeksforgeeks.org/python-foreground-extraction-in-an-image-using-grabcut-algorithm/
    mask = np.zeros(image.shape[:2], np.uint8)
    backgroundModel = np.zeros((1, 65), np.float64)
    foregroundModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, rectangle, 
                backgroundModel, foregroundModel,
                3, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8')
    image = image * mask2[:, :, np.newaxis]
    return image, mask2

def buffer_mask(_mask, buffer=3):
    mask_255 = _mask * 255
    mask_255_blur = cv2.blur(mask_255, (buffer, buffer))
    mask_255_blur[mask_255_blur>0]=1
    return mask_255_blur

def smooth_mask(_m, factor):
    _m_255 = _m * 255
    _m_255_blur = cv2.blur(_m_255, (factor, factor))
    _m_255_blur[_m_255_blur*2<=255] = 0
    _m_255_blur[_m_255_blur*2>255] = 1
    return _m_255_blur

def unique_colors(_im, return_counts=False):
    return np.unique(
        _im.reshape(-1, _im.shape[2]),
        axis=0,
        return_counts=return_counts
    )

def blur(_im, radius=5):
    return cv2.blur(_im, (radius, radius))

def invert(_im):
    return np.logical_not(_im)

def mask_image(_img, _mask):
    return _mask[:, :, np.newaxis]*_img

def mask_or(mask1, mask2):
    return np.maximum(mask1, mask2)