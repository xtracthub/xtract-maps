import numpy as np
import json
import cv2
from PIL import Image
from contouring import pre_process_image
from coordinate_extraction import pixel_to_coords_map, coord_to_abs, valid_pixel_map


def valid_contour(contour, img_dim, min_fraction=0.001, max_fraction=0.7):
    '''Checks if contour is a valid shape in the map by checking its area'''
    height, width = img_dim[:2]
    img_area = float(height * width)
    min_area, max_area = min_fraction * img_area, max_fraction * img_area
    area = cv2.contourArea(contour)
    if area < min_area or area > max_area:
        return False
    return True


def contour_approximation(contour, approximation=0.01):
    '''
    :param approximation: measures error distance between approx and original
    Returns an approximate polygon/contour for the contour given
    '''
    if not approximation:
        return contour
    epsilon = approximation * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx


def extract_borders(img, path_given=False, approximation=None, close=True, close_window_size=8, min_fraction=0.001, max_fraction=0.7, debug=False):
    '''
    Given an image, extracts contours from it, and returns those
    contours that fit the area bound criteria provided
    '''
    cv_img = cv2.imread(img) if path_given else np.array(img)
    cv_img = pre_process_image(cv_img, close=close, close_window_size=close_window_size, whiten=False)

    if debug:
        cv2.imwrite('pre_processed_image.jpg', cv_img)
        print('Written pre_processed_image.jpg')

    ret, thresh = cv2.threshold(cv_img, 127, 255, 0)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = [
            contour_approximation(x, approximation) for x in contours
            if valid_contour(x, cv_img.shape, min_fraction, max_fraction)
        ]

    return filtered_contours


def borders_to_coordinates(borders, pixel_to_coords, absolute=False):
    '''
    Maps a list of contours to a list of coordinate borders using the
    pixel_to_coords map provided, where each coordinate border is
    a list of (lat, lon) coordinate pairs
    :param absolute: if True, return -70 instead of [70, 'W']
    '''
    coord_borders = list()
    for border in borders:
        # OpenCV contours have an extra layer to get to each x,y pair
        border = np.array([p[0] for p in border])
        coord_border = [pixel_to_coords(x, y) for x, y in border]
        if absolute:
            coord_border = [
                (
                coord_to_abs(x[0], x[1], flip_lon=False, to_int=False),
                coord_to_abs(y[0], y[1], flip_lon=False, to_int=False)
                )
            for x, y in coord_border
            ]
        coord_borders.append(coord_border)

    return np.array(coord_borders)


def extract_borders_with_coordinates(img, path_given=False, approximation=None, absolute=False, debug=False):
    '''
    Given an image, generates a pixel_to_coords map using the
    coordinate_extraction module, then extracts pixel contours from the image,
    and then maps these contours to coordinate borders
    Note: The original image is left unchanged
    '''
    cv_img = cv2.imread(img) if path_given else np.array(img)
    try:
        pixel_to_coords = pixel_to_coords_map(cv_img, path_given=False, debug=debug)
        if debug:
            print('Generated pixel_to_coords map')
        borders = extract_borders(cv_img, path_given=False, approximation=approximation, debug=debug)
        return borders_to_coordinates(borders, pixel_to_coords, absolute=absolute)
    except RuntimeError as e:
        print(e)
        return None


if __name__ == '__main__':
    images = [
        # 'pub8_images/CAIBOX_2009_map.jpg',
        # 'pub8_images/GOMECC2_map.jpg',
        # 'pub8_images/EQNX_2015_map.jpg',
        # 'pub8_images/Marion_Dufresne_map_1991_1993.jpg',
        # 'pub8_images/P16S_2014_map.jpg',
        # 'pub8_images/Oscar_Dyson_map.jpg',
        # 'pub8_images/Bigelow2015_map.jpg',
        'pub8_images/A16S_2013_map.jpg'
        # 'pub8_images/woce_a25.gif',
        # 'pub8_images_2/pub8.oceans.save.SAVE.jpg'
        ]

    for image in images:
        cv_img = cv2.imread(image)
        borders = extract_borders_with_coordinates(cv_img, path_given=False, approximation=0.01, debug=False, absolute=True)
        print('For image', image, '\n')
        for i, border in enumerate(borders):
            print('Border', i)
            print(border)

    for image in images:
        cv_img = cv2.imread(image)
        borders = extract_borders(cv_img, approximation=0.01)
        cv_img = cv2.drawContours(cv_img, borders, -1, (0,0,255), 4)
        img = Image.fromarray(cv_img)
        img.show()
        # print(len(borders), 'borders found')
        # input('Press enter')
