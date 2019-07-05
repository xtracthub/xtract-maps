import numpy as np
import re
import cv2
import logging
from text_extraction import extract_text


# Setup for the debug logger
logging.basicConfig(format='%(asctime)s - %(filename)s - %(funcName)s - %('
                          'message)s', level=logging.DEBUG,
                    filename='debug.txt', filemode='w')
logger = logging.getLogger('logger')


def get_coordinates_from_text(box_to_text):
    """Parses each piece of text for a lat or lon coordinate.

    Parameter:
    box_to_text (dictionary(tuple : str)): A dictionary containing
    coordinates of a text box and the text from the text box.

    Return:
    coords (dictionary): Dictionary which maps each type of coordinate
    ('N', 'S', 'E', 'W') to a location-to-coordinate dictionary.
    """
    coords = {'N': {}, 'S': {}, 'E': {}, 'W': {}, 'other':{}}
    pattern = re.compile(r"(\d+)[^a-zA-Z]{,2}([NSEWnsew]?)")
    for loc, text in box_to_text.items():
        coords['other'][loc] = text
        text = text.strip().replace(' ', '')
        result = re.search(pattern=pattern, string=text)
        if result:
            groups = result.groups()
            if len(groups) == 2 and groups[1] and groups[1] in 'NSEWnsew':
                coords[groups[1].upper()][loc] = groups[0]
    return coords


def max_by_val(d, direc):
    """Finds the largest coordinate value of a specified direction.

    Parameters:
    d (dictionary): A direction-to-location-to-coordinate dictionary as
    returned by get_coordinates_from_text.
    direc (str): The direction you want to find the largest coordinate
    from ('N', 'E', 'S', 'W' ).

    Return:
    (tuple): 3-tuple of location, coordinate, direction.
    """
    m = max(d[direc].items(), key=lambda x: int(x[1]))
    return m[0], m[1], direc


def min_by_val(d, direc):
    """Finds the smallest coordinate value of a specified direction.

    Parameters:
    d (dictionary): A direction-to-location-to-coordinate dictionary as
    returned by get_coordinates_from_text.
    direc (str): The direction you want to find the smallest
    coordinate from ('N', 'E', 'S', 'W' ).

    Return:
    (tuple): 3-tuple of location, coordinate, direction.
    """
    m = min(d[direc].items(), key=lambda x: int(x[1]))
    return m[0], m[1], direc


def coord_to_abs(value, direction, flip_lon=False, to_int=False):
    """Converts a number and direction into a positive or negative
    integer (e.g. from (50, "N") and (100, "W") to 50 and -100
    respectively).

    Parameters:
    value (int): Value of direction.
    direction (str): Direction of value ('N', 'E', 'S', 'W')
    flip_lon (bool): If true, 'E' is treated as negative and 'W' is
    treated positive, otherwise 'E' is treated as positive and 'W' is
    treated as negative.
    to_int (bool): Whether to cast value as an int.

    Return:
    (float/int): Returns a positive or negative float/int, depending on
    whether to_int is true or false.
    """
    direction = direction.upper()
    cast = int if to_int else float
    positives = 'NW' if flip_lon else 'NE'
    if not (len(direction) == 1 and direction in 'NSEW'):
        return None
    return cast(value) if direction in positives else -1 * cast(value)


def coord_from_abs(value, orientation, flip_lon=False, to_int=False):
    """Converts a number and an orientation into a positive or
    negative integer (e.g. from (50, "N") and (100, "W") to 50 and
    -100 respectively).

    Parameters:
    value (int): Value of latitude or longitude.
    orientation (str): Orientation of value ('Latitude', 'Vertical',
    'Longitude', 'Horizontal').
    flip_lon (bool): If true, 'E' is treated as negative and 'W' is
    treated positive, otherwise 'E' is treated as positive and 'W' is
    treated as negative.
    to_int (bool): Whether to cast value as an int.

    Return:
    (tuple): 2-tuple of a positive value and a direction ('N', 'E', 'S', 'W').
    """
    orientation = orientation.upper()
    if to_int:
        value = int(value)
    if orientation in ['LAT', 'LATITUDE', 'VER', 'VERTICAL']:
        value, direction = (-1*value, 'S') if value < 0 else (value, 'N')
        if value > 90:
            value = 90
        return str(value), direction
    elif orientation in ['LONG', 'LON', 'LONGITUDE', 'HOR', 'HORIZONTAL']:
        neg, pos = ('E', 'W') if flip_lon else ('W', 'E')
        value, direction = (-1*value, neg) if value < 0 else (value, pos)
        if value > 180:
            value = 360 - value
            direction = neg if direction == pos else pos
        return str(value), direction
    else:
        return None


def any_coord_equal(coords1, coords2, orientation):
    """Checks whether the position or the corresponding x or y
    coordinate of 2 coordinate pairs is the same or not. Note: If
    both coordinates are empty, returns True.

    Parameters:
    coords1 (tuple): Tuple of the first coordinate in the form
    ((x,y), val, direc).
    coords2 (tuple): Tuple of the second coordinate in the form
    ((x,y), val, direc).
    orientation (str): The orientation to check whether corresponding
    coordinates are the same (e.g. 'Vertical' will check corresponding
    y-coordinates). Accepts 'Latitude', 'Vertical', 'Longitude',
    'Horizontal'.

    Returns:
    (bool): Whether coordinate pairs or corresponding coordinates are
    the same.
    """
    if coords1 == coords2:
        return True
    pos1, v1, d1 = coords1
    pos2, v2, d2 = coords2
    orientation = orientation.upper()
    if orientation in ['LAT', 'LATITUDE', 'VER', 'VERTICAL']:
        pos1, pos2 = pos1[1], pos2[1]   # vertical coordinate
    elif orientation in ['LONG', 'LON', 'LONGITUDE', 'HOR', 'HORIZONTAL']:
        pos1, pos2 = pos1[0], pos2[0]   # horizontal coordinate
    return pos1 == pos2 or v1 == v2


def valid_coordinate(coord):
    """Checks whether a coordinate is a valid coordinate.

    Parameter:
    coord (tuple): 2-tuple of latitude or longitude value and 'N', 'E',
    'S','W'.

    Returns:
    (bool): Whether coord is a valid coordinate.
    """
    val, direc = coord
    thresh = 90 if direc.upper() in 'NS' else 180
    val = float(val)
    return 0 <= val <= thresh


def valid_coordinate_pair(pair):
    """Checks whether a coordinate pair is valid or not.

    Parameter:
    pair (tuple): Tuple of a longitude and latitude (in that order).
    Longitude and latitude should be a 2-tuple of latitude or longitude
    value and 'N', 'E', 'S', 'W'.

    Return:
    (bool): Whether pair is a valid coordinate pair or not.
    """
    lon, lat = pair
    return all([
        lon[1].upper() in 'EW',
        lat[1].upper() in 'NS',
        valid_coordinate(lon),
        valid_coordinate(lat),
    ])


def valid_pixel_map(pixel_to_coords, img_dim, edge=20):
    """Naively check for whether a pixel_to_coords mapping is correct.
    Currently, only checks whether each image corner has valid
    coordinates.

    Parameters:
    pixel_to_coords (func): Function that maps a pixel location to a
    [longitude, latitude] pair (as returned by pixel_to_coords_map).
    img_dim (tuple): Dimensions of image that pixel_to_coords was
    extracted from.
    edge (int): Number of pixels from image border to consider extent
    of map.

    Return:
    (bool): Whether pixel_to_coords mapping is correct.
    """
    if not pixel_to_coords:
        return False
    test_coords = [
            [edge, edge],
            [img_dim[0]-edge, edge],
            [edge, img_dim[1]-edge],
            [img_dim[0]-edge, img_dim[1]-edge],
        ]
    test_coords = [pixel_to_coords(x,y) for (x,y) in test_coords]
    return all(map(valid_coordinate_pair, test_coords))


def get_coordinate_span_directly(coords, locations=False):
    """Gets the latitude and longitude span found in coords.

    Parameters:
    coords (dictionary): Dictionary which maps each type of coordinate
    ('N', 'S', 'E', 'W') to a location-to-coordinate dictionary
    (as returned by get_coordinates_from_text).
    locations (bool): Whether to return a location-value-direction tuple
    or a value-direction tuple.

    Return:
    (list (tuple)): A list of longitude and latitude. Longitude and
    latitude are location-value-direction tuples if locations is
    true, otherwise just a value-direction tuple.
    """
    lon, lat = [(), ()], [(), ()]
    if coords['S'] and coords['N']:
        lat[0] = max_by_val(coords, 'S')
        lat[1] = max_by_val(coords, 'N')
    elif coords['S']:
        lat[0] = min_by_val(coords, 'S')
        lat[1] = max_by_val(coords, 'S')
    elif coords['N']:
        lat[0] = min_by_val(coords, 'N')
        lat[1] = max_by_val(coords, 'N')
    if coords['E'] and coords['W']:
        east_min_x = min([x[0] for x in coords['E'].keys()])
        west_min_x = min([x[0] for x in coords['W'].keys()])
        if east_min_x < west_min_x:   # east to west from left to right
            lon[0] = min_by_val(coords, 'E')
            lon[1] = min_by_val(coords, 'W')
        else:
            lon[0] = max_by_val(coords, 'W')
            lon[1] = max_by_val(coords, 'E')
    elif coords['E']:
        lon[0] = min_by_val(coords, 'E')
        lon[1] = max_by_val(coords, 'E')
    elif coords['W']:
        lon[0] = min_by_val(coords, 'W')
        lon[1] = max_by_val(coords, 'W')

    if locations:
        return [lon, lat]
    else:
        lon = [lon[0][1:], lon[1][1:]]
        lat = [lat[0][1:], lat[1][1:]]
        return [lon, lat]


def pixel_to_coords_to_span(pixel_to_coords, img_dim, absolute=False, edge=20):
    """Naively returns the coordinate span of a map image.

    Parameters:
    pixel_to_coords (func): Function that returns a longitude and
    latitude pair given pixel coordinates from an image (as returned by
    pixel_to_coords_map).
    img_dim (tuple): 2-tuple of image dimensions.
    absolute (bool): Returns a positive/negative latitude and longitudes
    if true or a value-direction tuple if false.
    edge (int): Number of pixels from image border to consider extent of
    map.

    Return:
    (list): List of longitude maximum and minimum and latitude maximum
    and minimum.
    """
    height, width = img_dim[:2]
    lon_min, lat_min = pixel_to_coords(edge, height - edge)
    lon_max, lat_max = pixel_to_coords(width - edge, edge)
    if absolute:
        lon_min, lat_min, lon_max, lat_max = [
            coord_to_abs(lon_min[0], lon_min[1], flip_lon=False, to_int=False),
            coord_to_abs(lat_min[0], lat_min[1], flip_lon=False, to_int=False),
            coord_to_abs(lon_max[0], lon_max[1], flip_lon=False, to_int=False),
            coord_to_abs(lat_max[0], lat_max[1], flip_lon=False, to_int=False)
        ]
    return [[lon_min, lon_max], [lat_min, lat_max]]


def get_coordinate_span(coords, infer=True, img_dim=None, absolute=False):
    """Extrapolates the longitude and latitude span from coords.

    Parameters:
    coords dictionary): Dictionary which maps each type of coordinate
    ('N', 'S', 'E', 'W') to a location-to-coordinate dictionary (as
    returned by get_coordinates_from_text).
    infer (bool): If false, extracts directly, if true, infers expanse.
    edge (int): Number of pixels from image border to consider extent of
    map.

    Return:
    (list): List of longitude maximum and minimum and latitude
    maximum and minimum.
    """
    if not infer:
        return get_coordinate_span_directly(coords, locations=False)

    try:
        height, width = img_dim[:2]
    except TypeError:
        raise ValueError('Must provide img_dim when infer is set to True')

    pixel_to_coords = pixel_to_coords_map_helper(coords, to_int=True)
    return pixel_to_coords_to_span(pixel_to_coords, img_dim=(height, width),
                                   absolute=absolute)


def extract_coordinates(img, path_given=False, infer=True, padding=10,
                        dist_threshold=12, debug=False):
    """Finds the coordinate span of an image.
    Note: Works better with even number padding for some reason.

    Parameters:
    img (OpenCV image or file path): OpenCV image or file path to
    extract coordinates from.
    path_given (bool): Whether a file path is given in img.
    infer (bool): If false, extracts directly, if true, infers expanse.
    padding (int): Number of pixels of padding in img.
    dist_threshold (int): Number of pixels to consider boxes being too
    close.

    Return:
    (list): List of largest and smallest longitudes and latitudes.
    """
    cv_img = cv2.imread(img) if path_given else img
    box_to_text = extract_text(cv_img, debug=debug, padding=padding,
                               dist_threshold=dist_threshold)
    coords = get_coordinates_from_text(box_to_text)
    if debug:
        logger.debug(box_to_text)
        logger.debug(coords)
    return get_coordinate_span(coords, infer=infer, img_dim=img.shape)


def pixel_to_coords_map(img, path_given=False, padding=10, dist_threshold=12,
                        to_int=False, return_all_text=False, debug=False):
    """Given an image of a map, returns a function that maps a pixel
    location to a [longitude, latitude] pair.

    Parameters:
    img (OpenCV image or file path): OpenCV image or file path to
    extract coordinate span from.
    path_given (bool): Whether a file path is given in img.
    padding (int): Number of pixels of padding in img.
    dist_threshold (int): Number of pixels to consider boxes being too
    close.
    to_int (bool): Whether to turn values into integers.
    return_all_text (bool): If return_all_text is True, returns all
    coordinates extracted as well.

    Return:
    (func/tuple): If return_all_text is true, returns a 2-tuple of a
    function that returns a latitude and longitude given image
    coordinates and a dictionary of extracted coordinates, if false,
    it just returns the function.
    """
    cv_img = cv2.imread(img) if path_given else img
    box_to_text = extract_text(cv_img, debug=debug, padding=padding,
                               dist_threshold=dist_threshold)
    coords = get_coordinates_from_text(box_to_text)

    if debug:
        logger.debug(box_to_text)
        logger.debug(coords)
    if return_all_text:
        return pixel_to_coords_map_helper(coords, to_int=to_int,
                                          debug=debug), coords
    else:
        return pixel_to_coords_map_helper(coords, to_int=to_int, debug=debug)


def pixel_to_coords_map_helper(coords, to_int=False, debug=False):
    """Helper function for pixel_to_coords_map.

    Parameters:
    coords (dictionary): Dictionary which maps each type of coordinate
    ('N', 'S', 'E', 'W') to a location-to-coordinate dictionary (as
    returned by get_coordinates_from_text).
    to_int (bool): Whether to turn values into integers.

    Return:
    pixel_to_coords (func): Function that returns a latitude and
    longitude given a pixel coordinate from an image.
    """
    lon, lat = get_coordinate_span_directly(coords, locations=True)

    if any_coord_equal(lat[0], lat[1], 'LAT') or any_coord_equal(lon[0],
                                                                 lon[1], 'LON'):
        if debug:
            logger.debug('Number of valid coordinates found is not enough to '
                         'extrapolate a mapping from pixels to coordinates')
        return None

    def to_latitude(y):
        """Returns a latitude value for a y-coordinate from an image.

        Parameter:
        y (int):  y-coordinate from an image.

        Return:
        lat_y (float/int): Latitude value for y.
        """
        pos1, v1, d1 = lat[0]
        pos2, v2, d2 = lat[1]
        v1, v2 = coord_to_abs(v1, d1), coord_to_abs(v2, d2)
        y1, y2 = pos1[1], pos2[1]
        slope = abs(float(v2 - v1) / (y2 - y1))
        lat_y = coord_from_abs(v1 - slope * (y - y1), 'LAT', to_int=to_int)
        return lat_y

    def to_longitude(x):
        """Returns a longitude value for a x-coordinate from an image.

        Parameter:
        x (int):  x-coordinate from an image.

        Return:
        lon_x (float/int): Longitude value for x.
        """
        pos1, v1, d1 = lon[0]
        pos2, v2, d2 = lon[1]
        east_to_west = lon[0][2] == 'E' and lon[1][2] == 'W'
        v1, v2 = coord_to_abs(v1, d1, flip_lon=east_to_west), \
                 coord_to_abs(v2, d2, flip_lon=east_to_west)
        x1, x2 = pos1[0], pos2[0]
        delta_v = 360 - v2 + v1 if east_to_west else v2 - v1
        slope = abs(float(delta_v) / (x2 - x1))
        if east_to_west:
            lon_x = coord_from_abs(v2 - slope * (x - x2), 'LONG',
                                   flip_lon=east_to_west, to_int=to_int)
        else:
            lon_x = coord_from_abs(v1 + slope * (x - x1), 'LONG',
                                   flip_lon=east_to_west, to_int=to_int)
        return lon_x

    def pixel_to_coords(x, y=None):
        """Returns a longitude and latitude pair given an x and y
        coordinate from an image.

        Parameters:
        x (int/tuple): x-coordinate of an image or a 2-tuple of an x and
        y coordinate.
        y (int): y-coordinate of an image.

        Return:
        (tuple): 2-tuple of longitude and latitude for image
        coordinates.
        """
        try:
            x.__iter__
        except (SyntaxError, SyntaxWarning, AttributeError):
            return to_longitude(x), to_latitude(y)
        else:
            return to_longitude(x[0]), to_latitude(x[1])

    return pixel_to_coords


def pixel_to_coords_map_multiple_tries(img, path_given=False,
                                       return_all_text=False, debug=False):
    """Sequentially tries different padding and dist_threshold values to
    extract pixel_to_coords map, and returns the first valid map found.

    This exists because different maps have different spacing/format of
    text and one of the current two ways usually works for most
    images till now.

    Parameters:
    img (OpenCV image or file path): OpenCV image or file path..
    path_given (bool): Whether a file path is given in img.
    return_all_text (bool): If return_all_text is True, returns all
    coordinates extracted as well.

    Return:
    (func/tuple): If return_all_text is true, returns a 2-tuple of a
    function that returns a latitude and longitude given image
    coordinates and a dictionary of extracted coordinates, if false,
    it just returns the function.
    """
    cv_img = cv2.imread(img) if path_given else np.array(img)
    paddings = [20, 10]
    dist_thresholds = [12, 12]
    n = len(paddings)
    pixel_to_coords, coords = None, None
    for i in range(n):
        p, d = paddings[i], dist_thresholds[i]
        if debug:
            logger.debug(f'Trying with padding {p} and thresh {d}')
        pixel_to_coords = pixel_to_coords_map(cv_img, path_given=False,
                                              padding=p, dist_threshold=d,
                                              return_all_text=return_all_text,
                                              debug=debug)
        if return_all_text:
            pixel_to_coords, coords = pixel_to_coords

        if not pixel_to_coords:
            if debug:
                logger.debug('Failed with this padding and thresh')
            continue
        elif not valid_pixel_map(pixel_to_coords, cv_img.shape):
            if debug:
                logger.debug('Invalid map with this padding and thresh')
            continue
        else:
            break

    return (pixel_to_coords, coords) if return_all_text else pixel_to_coords
