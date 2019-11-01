import os
import cv2
import numpy as np
import random


def paste_over(occluder, image, alpha, center):
    """
    Pastes 'occluder' onto 'image' at a specified position, with alpha blending.

    The resulting image has the same shape as 'image' but contains 'occluder'
    (perhaps only partially, if it's put near the border).
    Locations outside the bounds of 'image' are handled as expected
    (only a part or none of 'occluder' becomes visible).

    Args:
        occluder: The image to be pasted onto 'image'. Its size can be arbitrary.
        image: The target image.
        alpha: A float (0.0, 1.0) image of the same size as 'occluder' controlling the alpha blending
            at each pixel. Large values mean more visibility for 'occluder'.
        center: coordinates in 'image' where the center of 'occluder' should be placed.

    Returns:
        An image of the same shape as 'image', with 'occluder' pasted onto it.
    """

    shape_occ = np.array(occluder.shape[:2])
    shape_image = np.array(image.shape[:2])

    center = np.round(center).astype(int)
    
    ideal_start_dst = center - shape_occ / 2
    ideal_end_dst = ideal_start_dst + shape_occ

    start_dst = np.maximum(ideal_start_dst, 0)
    end_dst = np.minimum(ideal_end_dst, shape_image)

    region_dst = image[start_dst[0]:end_dst[0], start_dst[1]:end_dst[1]]

    start_src = start_dst - ideal_start_dst
    end_src = shape_occ + (end_dst - ideal_end_dst)

    if alpha is None:
        alpha = np.ones(occluder.shape[:2], dtype=np.float32)

    if alpha.ndim < occluder.ndim:
        alpha = np.expand_dims(alpha, -1)

    alpha = alpha[start_src[0]:end_src[0], start_src[1]:end_src[1]]

    occluder = occluder[start_src[0]:end_src[0], start_src[1]:end_src[1]]

    image[start_dst[0]:end_dst[0], start_dst[1]:end_dst[1]] = (alpha * occluder + (1 - alpha) * region_dst)
    
    return image


def fetch_occluders(occ_idx, occ_path):
    occluder_file = os.path.join(occ_path, 'occluder_' + str(occ_idx) + '.npy')
    mask_file = os.path.join(occ_path, 'mask_' + str(occ_idx) + '.npy')

    occluder = np.load(occluder_file)
    mask = np.load(mask_file)

    return occluder, mask


def random_occlu(image, occ_count, occ_path):
    occ_idx = np.random.choice(occ_count)

    occluder, occ_mask = fetch_occluders(occ_idx, occ_path)

    dest_shape = np.random.uniform(0.4, 0.8) * np.array(occluder.shape[:2])
    dest_shape = tuple(np.round(dest_shape).astype(int))

    occluder = cv2.resize(occluder, dest_shape[::-1], interpolation = cv2.INTER_AREA)
    occ_mask = cv2.resize(occ_mask, dest_shape[::-1], interpolation = cv2.INTER_AREA)
    
    center = np.array(image.shape[:2]) * np.random.uniform(size = 2)

    return paste_over(occluder, image, alpha = occ_mask, center = center)


def random_erase(image):
    rand_color = np.random.randint(0, 256, size = 3)

    image_area = image.size / image.shape[-1]

    erase_area = np.random.uniform(0.1, 0.25) * image_area
    aspect_ratio = np.random.uniform(0.4, 2.5)

    erase_height = (erase_area * aspect_ratio) ** 0.5
    erase_width = (erase_area / aspect_ratio) ** 0.5

    erase_shape = np.array([erase_height, erase_width])

    erase_start = (np.array(image.shape[:2]) - erase_shape) * np.random.uniform(size = 2)
    erase_end = erase_start + erase_shape

    erase_start = np.round(erase_start).astype(int)
    erase_end = np.round(erase_end).astype(int)

    image[erase_start[0]:erase_end[0], erase_start[1]:erase_end[1]] = rand_color

    return image
