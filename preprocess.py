import cv2
import random


def resize(image):
    return cv2.resize(image, (299,299), interpolation=cv2.INTER_AREA)


def flip(image):
    if random.randrange(2) == 1:
        image = cv2.flip(image, 1)
    return image


def rotate(image):
    if random.randrange(2) == 1:
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)
        angles = [90, 180, 270]
        theta = angles[random.randrange(3)]
        M = cv2.getRotationMatrix2D(center, theta, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
    return image


def process_image(image):
    image = resize(image)
    if random.randrange(2) == 1:
        image = flip(image)
        image = rotate(image)
    return image
