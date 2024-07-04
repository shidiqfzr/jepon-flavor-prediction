import cv2
import numpy as np
from __future__ import division
from matplotlib import pyplot as plt


green = (0, 255, 0)

def show(image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image, interpolation='nearest')

def overlay_mask(mask, image):
    rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    img = cv2.addWeighted(rgb_mask, 0.5, image, 0.5, 0)
    return img

def find_biggest_contour(image):
    image = image.copy()
    contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [biggest_contour], -1, 255, -1)
    return biggest_contour, mask

def circle_contour(image, contour):
    image_with_ellipse = image.copy()
    ellipse = cv2.fitEllipse(contour)
    cv2.ellipse(image_with_ellipse, ellipse, green, 2, cv2.LINE_AA)
    return image_with_ellipse

def find_orange(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    max_dimension = max(image.shape)
    scale = 700/max_dimension
    image = cv2.resize(image, None, fx=scale, fy=scale)
    image_blur = cv2.GaussianBlur(image, (7, 7), 0)
    image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)

    min_ora = np.array([10, 50, 50])
    max_ora = np.array([30, 255, 255])
    mask1 = cv2.inRange(image_blur_hsv, min_ora, max_ora)

    min_ora2 = np.array([170, 50, 50])
    max_ora2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(image_blur_hsv, min_ora2, max_ora2)
    
    mask = mask1 + mask2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

    big_orange_contour, mask_oranges = find_biggest_contour(mask_clean)
    overlay = overlay_mask(mask_clean, image)
    circled = circle_contour(overlay, big_orange_contour)
    show(circled)
    bgr = cv2.cvtColor(circled, cv2.COLOR_RGB2BGR)
    return bgr

image = cv2.imread(r'sliced-whole-oranges.jpg')
result = find_orange(image)
cv2.imwrite(r'sliced-whole-oranges_result.jpg', result)