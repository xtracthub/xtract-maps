import numpy as np
import cv2
import logging
from PIL import Image
import pytesseract
from .contouring import isolate_text_boxes, boxes_to_contours


# Setup for the debug logger
logging.basicConfig(format='%(asctime)s - %(filename)s - %(funcName)s - %('
                          'message)s', level=logging.DEBUG,
                    filename='debug.txt', filemode='w')
logger = logging.getLogger('logger')


def get_partial_image(box, cv_img):
    """Returns new array with only part of original image.

    Parameters:
    box (list(tuple)): List of box corner coordinates.
    cv_img (OpenCV image): OpenCV image to take a part of.

    Return:
    (numpy array): New array of box from cv_img.
    """
    min_x, min_y = box[0]
    max_x, max_y = box[1]
    return np.array(cv_img[min_y : max_y, min_x : max_x])


def get_text(img):
    """Returns text from a PIL image.

    Parameter:
    img (PIL image): PIL image to get text from.

    Return:
    (str): String of text form img.
    """
    return pytesseract.image_to_string(img)


def get_text_from_boxes(boxes, cv_img):
    """Returns a location-to-text dictionary of text found inside the
    boxes in the image (where location is considered to be the center of
    the box).

    Parameters:
    boxes (list(box)): List of box coordinates.
    cv_img (OpenCV image): OpenCV image to get text from.

    Return:
    box_to_text (dictionary(tuple : str)): Dictionary containing
    coordinates of center of box in boxes and text from box.
    """
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
    """Isolates text boxes found in the OpenCV image, and then uses
    Tesseract to extract text from each box, returning a
    location-to-text dictionary.

    Parameters:
    img (OpenCV image): OpenCV image to extract text from.
    padding (int): Number of pixels of paddinf on img.
    dist_threshold (int): Threshold of number of pixels to determine
    whether boxes are too close.

    Return:
    box_to_test (dictionary(tuple : str)): Dictionary containing
    coordinates of center of text boxes and text from text boxes.
    """
    text_boxes = isolate_text_boxes(img, padding=padding,
                                    dist_threshold=dist_threshold)

    if debug:
        img_temp = np.array(img)
        img_temp = cv2.drawContours(img_temp, boxes_to_contours(text_boxes),
                                    -1, (0,0,255), 2)
        output = 'image_with_text_boxes.jpg'
        cv2.imwrite(output, img_temp)
        logger.debug(f'Written file {output}')

    box_to_text = get_text_from_boxes(text_boxes,
                                      cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return box_to_text

