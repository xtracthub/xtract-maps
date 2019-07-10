import numpy as np
from math import sqrt
from itertools import combinations
import cv2
from unionfind import UnionFind


def valid_image(img):
    """Checks whether the image can be opened by OpenCV.

    Parameters:
    img (Image path or OpenCV image): Path of image or OpenCV image.

    Return:
    (bool): Whether img can be opened as an OpenCV image or if it is
    an OpenCV image.
    """
    if isinstance(img, str):
        img = cv2.imread(img)
        return img is not None
    elif isinstance(img, np.ndarray):
        return img.any() and img.ndim in [2,3]
    return False


def enclosing_box(contour, img_dim=None, padding=0):
    """Finds the enclosing rectangle for a given list of 2D points.
    Note: a rectangle/box is represented by 2 points (opposite corners)
    in this code, unless otherwise specified.

    Parameters:
    contour (list(tuple)): A list of boundary points for a contour.
    img_dim (tuple(int)): A set of dimensions of contour (only needed
    if padding is greater than 0.
    padding (int): Number of pixels of padding on contour.

    Return:
    (numpy array): Numpy array containing coordinates of top left and
    bottom right points of enclosing rectangle.
    """
    min_x, min_y, w, h = cv2.boundingRect(np.array(contour))
    max_x, max_y = min_x + w, min_y + h

    if padding:
        try:
            height, width = img_dim
        except Exception as e:
            raise TypeError('When padding is non-zero, '
                            'img_dim must be provided')
            return
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - 2 * padding)
        max_x = min(width, max_x + padding)
        max_y = min(height, max_y + 2 * padding)

    return np.array([[min_x, min_y], [max_x, max_y]])


def boxes_to_contours(boxes):
    """Given a list of 2-point boxes (as returned by enclosing_box),
    returns a list of 4-point rectangles which can then be drawn using
    cv2.drawContours.

    Parameter:
    boxes (list(numpy array): List of numpy arrays containing
    enclosing box coordinates (as returned by enclosing_box).

    Return:
    rects (list(numpy array)): List of numpy arrays containing all
    points of enclosing box coordinates.
    """
    rects = [
        np.array([
            [box[0][0], box[0][1]],
            [box[0][0], box[1][1]],
            [box[1][0], box[1][1]],
            [box[1][0], box[0][1]]
        ])
        for box in boxes
    ]
    return rects


def dist(p, q):
    """Euclidean distance between two points.

    Parameters:
    p (list(int)): Coordinates for a point.
    q (list(int)): Coordinates for a point.

    Return:
    (float): Distance between points p and q
    """
    return sqrt((p[0] - q[0])**2 + (p[1] - q[1])**2)


def closest_distance(box1, box2):
    """Finds the closest distance between two rectangles/boxes.

    This distance is defined as the shortest distance between any 2
    points on the 2 boxes, and is hence 0 in case of an intersection.

    Parameters:
    box1 (numpy array): Numpy array of coordinates of a box as
    returned by enclosing_box.
    box2 (numpy array): Numpy array of coordinates of a box as
    returned by enclosing_box.

    Return:
    (float): Closest distance between box1 and box2. Returns 0 in the
    case of an intersection.
    """
    min_x_1, min_y_1 = box1[0]
    max_x_1, max_y_1 = box1[1]
    min_x_2, min_y_2 = box2[0]
    max_x_2, max_y_2 = box2[1]

    # All relations of the form: 'box1 <relation> box2',
    # e.g., 'box1 above box2'
    above = max_y_1 < min_y_2
    below = min_y_1 > max_y_2
    left = max_x_1 < min_x_2
    right = min_x_1 > max_x_2

    if above and left:
        return dist((max_x_1, max_y_1), (min_x_2, min_y_2))
    elif above and right:
        return dist((min_x_1, max_y_1), (max_x_2, min_y_2))
    elif below and left:
        return dist((max_x_1, min_y_1), (min_x_2, max_y_2))
    elif below and right:
        return dist((min_x_1, min_y_1), (max_x_2, max_y_2))
    elif above:
        return min_y_2 - max_y_1
    elif below:
        return min_y_1 - max_y_2
    elif left:
        return min_x_2 - max_x_1
    elif right:
        return min_x_1 - max_x_2
    else:    # intersect
        return 0


def fits_criteria_contour(box, img_dim):
    """Checks whether a box is too big or small in respect to the
    image dimensions.

    Parameters:
    box (numpy array): Numpy array of coordinates of a box as
    returned by enclosing_box.
    img_dim (tuple(int)): Tuple of image dimensions.

    Return:
    (bool): Whether box is too big or small in respect to img_dim.
    """
    height, width = img_dim
    min_x, min_y = box[0]
    max_x, max_y = box[1]

    hor_size = max_x - min_x
    ver_size = max_y - min_y

    if float(ver_size) / height > 0.05 or float(hor_size) / width > 0.05:
        return False
    if hor_size > 100 or hor_size < 5:
        return False
    if ver_size > 100 or ver_size < 2:
        return False
    if ver_size * hor_size < 100:
        return False
    return True


def fits_criteria_box(box):
    """Used to filter out boxes which are not likely to be text
    boxes.

    Parameter:
    box (numpy array): Numpy array of coordinates of a box as
    returned by enclosing_box.

    Return:
    (bool): Whether box is likely to be a text box or not.
    """
    min_x, min_y = box[0]
    max_x, max_y = box[1]

    hor_size = max_x - min_x
    ver_size = max_y - min_y
    if ver_size > 1.5 * hor_size:
        return False
    if ver_size * hor_size < 200:
        return False
    return True


def combine_boxes(boxes, img_dim, dist_threshold=15, padding=0):
    """Uses UnionFind to group close-by contours into boxes (disjoint
    connected components).

    Parameters:
    boxes (list(numpy array)): List of numpy arrays of coordinates of a
    box as returned by enclosing_box.
    img_dim (tuple(int)): Tuple of image dimensions.
    dis_threshold (int): Threshold of number of pixels to determine
    whether boxes are too close.
    padding (int): Number of pixels that are padding.

    Return:
    (numpy array): Numpy array of combined boxes.
    """
    n = len(boxes)
    uf = UnionFind(n)

    for i, j in combinations(range(n), 2):
        if closest_distance(boxes[i], boxes[j]) < dist_threshold:
            uf.union(i, j)

    box_groups = [[box for i in group for box in boxes[i]]
                  for group in uf.groups()]
    combined_boxes = [enclosing_box(group, img_dim, padding=padding)
                      for group in box_groups]
    filtered_boxes = [x for x in combined_boxes if fits_criteria_box(x)]
    return np.array(filtered_boxes)


def pre_process_image(img, close=True, whiten=True, white_thresh=50):
    """Converts OpenCV image to grayscale, and can apply
    a morphological filter and a whitening filter.
    Note: The original image is left unchanged.

    Parameters:
    img (OpenCV image): Image to be processed.
    close (bool): Whether to close holes and imperfections in img.
    whiten (bool): Whether to whiten img.
    white_thresh (int): Threshold for whitening img. Values
    above white_thresh become white, values below become black.

    Return:
    edited_img (OpenCV image): Edited version of img with filters
    applied.
    """
    edited_img = np.array(img)
    edited_img = cv2.cvtColor(edited_img, cv2.COLOR_BGR2GRAY)

    if close:
        kernel = np.ones((2, 2), np.uint8)
        edited_img = cv2.morphologyEx(edited_img, cv2.MORPH_CLOSE, kernel)
    if whiten:
        for i in range(edited_img.shape[0]):
            for j in range(edited_img.shape[1]):
                if edited_img[i][j] > white_thresh:
                    edited_img[i][j] = 255
                else:
                    edited_img[i][j] = 0

    return edited_img


def isolate_text_boxes(img, path_given=False, close=True, whiten=False,
                       padding=10, dist_threshold=12, return_contours=False):
    """Uses contour finding and grouping to isolate potential text
    boxes in the image. Note: The original image is left unchanged.

    Parameters:
    img (Image path or OpenCV file): Path to an image file or an
    OpenCV image.
    path_given (bool): Whether a file path was given to img.
    close (bool): Whether you want to close holes and imperfections
    in img.
    whiten (bool): Whether you want to whiten img.
    padding (int): Number of pixels of padding on img.
    dist_threshold (int): Threshold of number of pixels to determine
    whether boxes are too close.
    return_contours (bool): Whether to return all coordinates for all
    contour boxes.

    Return:
    text_boxes (numpy array): Returns a numpy array of text box
    coordinates. If return_contours is true, coordinates of all four
    corners are returned, if false, only two are returned.
    """
    img = cv2.imread(img) if path_given else img

    img_mod = pre_process_image(img, close=close, whiten=whiten)
    img_dim = img_mod.shape

    ret, thresh = cv2.threshold(img_mod, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = [x for x in map(lambda x: enclosing_box(x, img_dim),
                                        contours)
                         if fits_criteria_contour(x, img_dim)]

    text_boxes = combine_boxes(filtered_contours, img_dim, padding=padding,
                               dist_threshold=dist_threshold)

    if return_contours:
        text_boxes = boxes_to_contours(text_boxes)

    return text_boxes

