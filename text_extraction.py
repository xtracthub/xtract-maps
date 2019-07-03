import numpy as np
import cv2
from PIL import Image
import pytesseract

from contouring import isolate_text_boxes, boxes_to_contours


def get_partial_image(box, cv_img):
    '''Returns new array with only part of original image'''
    min_x, min_y = box[0]
    max_x, max_y = box[1]
    return np.array(cv_img[min_y : max_y, min_x : max_x])


def get_text(img):
    ''':param img: a PIL Image object'''
    return pytesseract.image_to_string(img)


def get_text_from_boxes(boxes, cv_img):
    '''
    Returns a location-to-text dictionary of text found inside the boxes
    in the image (where location is considered to be the center of the box)
    '''
    box_to_text = dict()
    for box in boxes:
        partial_img = get_partial_image(box, cv_img)
        partial_img = Image.fromarray(partial_img)
        location = (np.amin(box, axis=0) + np.amax(box, axis=0)) / 2
        location = tuple(location.astype(int))
        text = get_text(partial_img)
        if text:
            box_to_text[location] = text
    return box_to_text


def extract_text(img, padding=10, dist_threshold=12, debug=False):
    '''
    Isolates text boxes found in the OpenCV image, and then uses Tesseract to
    extract text from each box, returning a location-to-text dictionary
    '''
    text_boxes = isolate_text_boxes(img, padding=padding, dist_threshold=dist_threshold)

    if debug:
        img_temp = np.array(img)
        img_temp = cv2.drawContours(img_temp, boxes_to_contours(text_boxes), -1, (0,0,255), 2)
        output = 'image_with_text_boxes.jpg'
        cv2.imwrite(output, img_temp)
        print('Written file:', output)

    box_to_text = get_text_from_boxes(text_boxes, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return box_to_text
