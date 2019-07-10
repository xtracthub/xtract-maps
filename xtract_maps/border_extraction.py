import numpy as np
import cv2
import logging
from .contouring import pre_process_image
from .coordinate_extraction import pixel_to_coords_map, coord_to_abs


# Setup for the debug logger
logging.basicConfig(format='%(asctime)s - %(filename)s - %(funcName)s - %('
                          'message)s', level=logging.DEBUG,
                    filename='debug.txt', filemode='w')
logger = logging.getLogger('logger')


def valid_contour(contour, img_dim, min_fraction=0.001, max_fraction=0.7):
    """Checks if contour is a valid shape in the map by checking its
    area.

    Parameters:
    contour (list(tuple)): List of boundary points for a contour.
    img_dim (tuple): 2-tuple of image dimensions.
    min_fraction (float): Minimum fraction of image dimensions that
    contour's area has to be greater than.
    max_fraction (float): Maximum fraction of image dimensions that
    contour's area has to be less than.
    """
    height, width = img_dim[:2]
    img_area = float(height * width)
    min_area, max_area = min_fraction * img_area, max_fraction * img_area
    area = cv2.contourArea(contour)
    if area < min_area or area > max_area:
        return False
    return True


def contour_approximation(contour, approximation=0.01):
    """Returns an approximate polygon/contour for the contour given.

    contour (list(tuple)): A list of boundary points for a contour.
    approximation (float): Error distance between approximation and
    original.

    Return:
    approx (numpy array): Array of points for contour approximation.
    """
    if not approximation:
        return contour
    epsilon = approximation * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx


def extract_borders(img, path_given=False, approximation=None, close=True,
                    min_fraction=0.001, max_fraction=0.7, debug=False):
    """Given an image, extracts contours from it, and returns those
    contours that fit the area bound criteria provided.

    Parameters:
    img (OpenCV image or file path): OpenCV image or file path to
    extract borders from.
    path_given (bool): Whether a file path is given in img.
    approximation (float): Error distance between approximation and
    original contour.
    close (bool): Whether to close holes and imperfections in img.
    min_fraction (float): Minimum fraction of img that
    contour's area has to be greater than.
    max_fraction (float): Maximum fraction of img that
    contour's area has to be less than.

    Return:
    (list): List of approximated contours that fit the area bound
    criteria.
    """
    cv_img = cv2.imread(img) if path_given else np.array(img)
    cv_img = pre_process_image(cv_img, close=close, whiten=False)

    if debug:
        cv2.imwrite('pre_processed_image.jpg', cv_img)
        logger.debug('Written pre_processed_image.jpg')

    ret, thresh = cv2.threshold(cv_img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = [
            contour_approximation(x, approximation) for x in contours
            if valid_contour(x, cv_img.shape, min_fraction, max_fraction)
        ]

    return filtered_contours


def borders_to_coordinates(borders, pixel_to_coords, absolute=False):
    """Maps a list of contours to a list of coordinate borders.

    Parameters:
    borders (list): List of contours.
    pixel_to_coords (func): Function that maps pixel coordinates to
    latitudes and longitudes.
    absolute (bool): Returns positive or negative floats if true,
    returns a value-direction tuple if false.

    Return:
    (numpy array): Numpy array of latitudes and longitudes of each
    border.
    """
    coord_borders = list()
    for border in borders:
        # OpenCV contours have an extra layer to get to each x,y pair
        border = np.array([p[0] for p in border])
        coord_border = [pixel_to_coords(x, y) for x, y in border]
        if absolute:
            coord_border = [
                (coord_to_abs(x[0], x[1], flip_lon=False, to_int=False),
                 coord_to_abs(y[0], y[1], flip_lon=False, to_int=False))
                for x, y in coord_border
            ]
        coord_borders.append(coord_border)

    return np.array(coord_borders)


def extract_borders_with_coordinates(img, path_given=False, approximation=None,
                                     absolute=False, debug=False):
    """Given an image, generates a pixel_to_coords map using the
    coordinate_extraction module, then extracts pixel contours from the
    image, and then maps these contours to coordinate borders.
    Note: The original image is left unchanged.

    Parameters:
    img (OpenCV image or file path): OpenCV image or file path to
    extract borders from.
    path_given (bool): Whether a file path is given in img.
    approximation (float): Error distance between approximation and
    original contour.
    absolute (bool): Returns positive or negative floats if true,
    returns a value-direction tuple if false.

    Return:
    (numpy array): Numpy array of latitudes and longitudes of each
    border.
    """
    cv_img = cv2.imread(img) if path_given else np.array(img)

    try:
        pixel_to_coords = pixel_to_coords_map(cv_img, path_given=False,
                                              debug=debug)
        if debug:
            logger.debug('Generated pixel_to_coords map')
        borders = extract_borders(cv_img, path_given=False,
                                  approximation=approximation, debug=debug)
        return borders_to_coordinates(borders, pixel_to_coords,
                                      absolute=absolute)
    except RuntimeError as e:
        logger.error(exc_info=True)
        return None

# Test code, please ignore

# if __name__ == '__main__':
#     images = [
#         # 'pub8_images/CAIBOX_2009_map.jpg',
#         # 'pub8_images/GOMECC2_map.jpg',
#         # 'pub8_images/EQNX_2015_map.jpg',
#         # 'pub8_images/Marion_Dufresne_map_1991_1993.jpg',
#         # 'pub8_images/P16S_2014_map.jpg',
#         'Oscar_Dyson_map.jpg'
#         # 'pub8_images/Bigelow2015_map.jpg',
#         #'testpic.png'
#         # 'pub8_images/woce_a25.gif',
#         # 'pub8_images_2/pub8.oceans.save.SAVE.jpg'
#         ]
#
#     for image in images:
#         cv_img = cv2.imread(image)
#         borders = extract_borders_with_coordinates(cv_img, path_given=False,
#                                                    approximation=0.01,
#                                                    debug=False, absolute=True)
#         print('For image', image, '\n')
#         for i, border in enumerate(borders):
#             print('Border', i)
#             print(border)
#
#     for image in images:
#         cv_img = cv2.imread(image)
#         borders = extract_borders(cv_img, approximation=0.01)
#         cv_img = cv2.drawContours(cv_img, borders, -1, (0,0,255), 4)
#         img = Image.fromarray(cv_img)
#         img.show()
