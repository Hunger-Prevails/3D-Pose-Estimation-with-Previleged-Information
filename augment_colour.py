import cv2
import numpy as np
import random


def augment_brightness(image, space):
    if space != 'rgb':
        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

    image += np.random.uniform(-0.125, 0.125)

    return np.clip(image, 0, 1), 'rgb'


def augment_contrast(image, space):
    if space != 'rgb':
        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

    image -= 0.5
    image *= np.random.uniform(0.8, 1.25)
    image += 0.5
    
    return np.clip(image, 0, 1), 'rgb'


def augment_hue(image, space):
    if space != 'hsv':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    image[:, :, 0] += np.random.uniform(-18, 18)

    image[:, :, 0][image[:, :, 0] < 0] += 360
    image[:, :, 0][360 <= image[:, :, 0]] -= 360
    
    return image, 'hsv'


def augment_saturation(image, space):
    if space != 'hsv':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    image[:, :, 1] *= np.random.uniform(0.8, 1.25)
    image[:, :, 1] = np.clip(image[:, :, 1], 0, 1)

    return image, 'hsv'


def random_color(image):
    '''
    performs random colour augmentation for the given image

    Args:
        image: 3-channel rgb image in range [0, 256)
    '''
    dest = (image / 255.0).astype(np.float32)

    augment_funcs = [augment_brightness, augment_contrast, augment_hue, augment_saturation]

    colorspace = 'rgb'

    for func in augment_funcs:
        dest, colorspace = func(dest, colorspace)

    if colorspace != 'rgb':
        dest = cv2.cvtColor(dest, cv2.COLOR_HSV2RGB)

    return (dest * 255).astype(np.uint8)
