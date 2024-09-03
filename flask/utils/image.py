import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans

#https://stackoverflow.com/questions/30483886/kmeans-clustering-on-different-distance-function-in-lab-space


def load_image(_i_path):
    return cv2.imread(_i_path)


def natural_color(_i):
    return cv2.cvtColor(_i, cv2.COLOR_BGR2RGB)


def to_lab_color(_i):
    return cv2.cvtColor(_i, cv2.COLOR_BGR2LAB)


def from_lab_color(_i):
    return cv2.cvtColor(_i, cv2.COLOR_LAB2BGR)


def bgr_color(_i):
    return cv2.cvtColor(_i, cv2.COLOR_RGB2BGR)


def display_image(_i):
    plt.figure(figsize=(5.2, 3.9)) #6, 4.5 # 4.8, 3.6
    plt.imshow(_i)
    plt.colorbar()
    plt.show()


def display_natural(_i):
    _i = natural_color(_i)
    display_image(_i)


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


def blur(_im, radius=5):
    return cv2.blur(_im, (radius, radius))


def invert(_im):
    return np.logical_not(_im)


def mask_image(_img, _mask):
    return _mask[:, :, np.newaxis]*_img


def mask_or(mask1, mask2):
    return np.maximum(mask1, mask2)


def kmeans_cluster_masked_image(image, mask, k):
    """
    Applies K-means clustering to a masked image.

    Args:
        image (numpy.ndarray): The input image.
        mask (numpy.ndarray): A binary mask where 1 indicates pixels to cluster.
        k (int): The number of clusters.

    Returns:
        numpy.ndarray: The clustered image.
    """

    # Convert image to CIELAB color space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Apply mask to the image
    masked_image = image[mask==255]
    
    # Reshape the masked image for K-means
    masked_image = masked_image.reshape((-1, 3))

    # Apply K-means clustering
    kmeans = MiniBatchKMeans(n_clusters=k).fit(masked_image)
    labels = kmeans.labels_
    
    # Create the clustered image
    clustered_image = np.zeros_like(image)
    clustered_image[mask == 255] = kmeans.cluster_centers_[labels]
    
    # Convert back to BGR from lab
    clustered_image = cv2.cvtColor(clustered_image, cv2.COLOR_LAB2BGR)
    clustered_image[mask == 0] = 0

    return clustered_image


def kmeans_cielab(image, k):
    
    # Convert image to CIELAB color space
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Reshape the image for K-means
    pixels = image_lab.reshape((-1, 3))

    # Apply K-means clustering
    kmeans = MiniBatchKMeans(n_clusters=k)
    kmeans.fit(pixels)

    # Get cluster centers and labels
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Reconstruct the segmented image
    segmented_image = centers[labels].reshape(image.shape)
    segmented_image = np.uint8(segmented_image)

    return segmented_image, centers


def enhance_color_differences(image, alpha = 1.5):
    """Enhances color differences in an image using HSV color space."""

    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Increase saturation
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * alpha, 0, 255).astype(np.uint8)

    # Convert back to BGR
    enhanced_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return enhanced_image


def brighten_image(image, alpha=1.5, beta=10):
    """Brightens an image using alpha (contrast) and beta (brightness)."""

    new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return new_image


def raise_black_point(image, alpha=1.1, beta=-15):
    """Raises the black point of a color image using cv2.

    Args:
        image (numpy.ndarray): The input image.
        alpha (float): Contrast control (1.0 for no change).
        beta (int): Brightness control (0 for no change).

    Returns:
        numpy.ndarray: The adjusted image.
    """

    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image


def increase_vibrance(img, alpha=1.25, beta=20):
    """
    Increases the vibrance of an image using alpha (contrast) and beta (brightness).
    Note: alpha should be <= 2
    """

    # Convert image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Scale the saturation channel (V)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * alpha, 0, 255).astype(np.uint8)

    # Convert back to BGR
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Adjust brightness
    result = cv2.convertScaleAbs(result, alpha=1, beta=beta)

    return result
