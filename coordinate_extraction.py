import numpy as np
import re
import cv2
from text_extraction import extract_text


def get_coordinates_from_text(box_to_text):
    '''
    :param box_to_text: a location-to-text dictionary
    Parses each piece of text for a lat or lon coordinate
    Returns a dictionary which maps each type of coordinate
    ('N', 'S', 'E', 'W') to a location-to-coordinate dictionary
    '''
    coords = {'N': {}, 'S': {}, 'E': {}, 'W': {}, 'other':{}}
    pattern = re.compile(r"(\d+)[^a-zA-Z]{,2}([NSEWnsew]?)")
    for loc, text in box_to_text.items():
        coords['other'][loc] = text
        text = text.strip().replace(' ','')
        result = re.search(pattern=pattern, string=text)
        if result:
            groups = result.groups()
            if len(groups) == 2 and groups[1] and groups[1] in 'NSEWnsew':
                coords[groups[1].upper()][loc] = groups[0]
    return coords


def max_by_val(d, dir):
    '''
    :param d: a direction-to-location-to-coordinate dictionary as returned
                 by get_coordinates_from_text
    :param dir: a direction out of ('N', 'S', 'E', 'W')
    Returns a 3-tuple of location, coordinate, direction (the max coordinate)
    '''
    m = max(d[dir].items(), key=lambda x: int(x[1]))
    return m[0], m[1], dir


def min_by_val(d, dir):
    '''
    :param d: a direction-to-location-to-coordinate dictionary as returned
                 by get_coordinates_from_text
    :param dir: a direction out of ('N', 'S', 'E', 'W')
    Returns a 3-tuple of location, coordinate, direction (the min coordinate)
    '''
    m = min(d[dir].items(), key=lambda x: int(x[1]))
    return m[0], m[1], dir


def coord_to_abs(value, direction, flip_lon=False, to_int=False):
    '''
    Converts coordinates from (50, "N") and (100, "W") to 50 and -100 resp.
    If flip_lon is True, treats "W" as positive instead of "E"
    '''
    direction = direction.upper()
    cast = int if to_int else float
    positives = 'NW' if flip_lon else 'NE'
    if not (len(direction) == 1 and direction in 'NSEW'):
        return None
    return cast(value) if direction in positives else -1 * cast(value)


def coord_from_abs(value, orientation, flip_lon=False, to_int=False):
    '''
    Converts coordinates from 50 and -100 to (50, "N") and (100, "W") resp.
    If flip_lon is True, treats "W" as positive instead of "E"
    '''
    orientation = orientation.upper()
    if to_int: value = int(value)
    mult = 1 if flip_lon else -1
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
    '''
    Checks whether the position or the corresponding coordinate
    of 2 coordinates pairs is the same or not
    Each coord should be of the form: ((x,y), val, dir)
    Note: If both coordinates are empty, returns True
    '''
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
    '''Simple check for whether a coordinate is valid or not'''
    val, dir = coord
    thresh = 90 if dir.upper() in 'NS' else 180
    val = float(val)
    return val >= 0 and val <= thresh


def valid_coordinate_pair(pair):
    '''Simple check for whether a coordinate pair is valid or not'''
    lon, lat = pair
    return all([
        lon[1].upper() in 'EW',
        lat[1].upper() in 'NS',
        valid_coordinate(lon),
        valid_coordinate(lat),
    ])


def valid_pixel_map(pixel_to_coords, img_dim, edge=20):
    '''
    Very naive check for whether a pixel_to_coords mapping is correct
    Currently, only checks whether each image corner has valid coordinates
    '''
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
    '''
    Gets the latitude and longitude span found in coords (as from
    get_coordinates_from_text), returns a [lon, lat] pair, where
    lon and lat are each a pair of coordinates
    Each coordinate is a location-value-direction tuple if locations
    is True, otherwise a value-direction tuple
    '''
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
    '''
    Naively returns the coordinate span of a map image given
    a pixel_to_coords map and the dimensions of the image
    '''
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


def get_coordinate_span(coords, infer=True, img_dim=None, absolute=False, edge=20):
    '''
    :param infer: whether to infer expanse or simply extract directly
    :param edge: distance from image border to consider extent of map
    Extrapolates the longitude and latitude span found in coords (as from
    get_coordinates_from_text)
    Returns a [lon, lat] pair where each lon and lat is a pair of coordinates,
    and each coordinate is a value-direction pair
    '''
    if not infer:
        return get_coordinate_span_directly(coords, locations=False)

    try:
        height, width = img_dim[:2]
    except TypeError:
        raise ValueError('Must provide img_dim when infer is set to True')
        return None

    pixel_to_coords = pixel_to_coords_map_helper(coords, to_int=True)
    return pixel_to_coords_to_span(pixel_to_coords, img_dim=(height, width), absolute=absolute)


def extract_coordinates(img, path_given=False, infer=True, padding=10, dist_threshold=12, debug=False):
    '''
    Given an image, first extracts text using the text_extraction module,
    then gets the coordinate span of the map in the image
    Note: Tesseract works better with even number padding for some reason
    '''
    cv_img = cv2.imread(img) if path_given else img
    box_to_text = extract_text(cv_img, debug=debug, padding=padding, dist_threshold=dist_threshold)
    coords = get_coordinates_from_text(box_to_text)
    if debug:
        print(box_to_text)
        print(coords)
    return get_coordinate_span(coords, infer=infer, img_dim=img.shape)


def pixel_to_coords_map(img, path_given=False, padding=10, dist_threshold=12, to_int=False, return_all_text=False, debug=False):
    '''
    Given an image of a map, returns a function that maps a pixel
    location to a [longitude, latitude] pair
    If return_all_text is True, returns all coordinates extracted as well,
    inside a tuple of (pixel_to_coords function, coords) where coords is a
    dictionary as returned from get_coordinates_from_text
    '''
    cv_img = cv2.imread(img) if path_given else img
    box_to_text = extract_text(cv_img, debug=debug, padding=padding, dist_threshold=dist_threshold)
    coords = get_coordinates_from_text(box_to_text)

    if debug:
        print(box_to_text)
        print(coords)
    if return_all_text:
        return pixel_to_coords_map_helper(coords, to_int=to_int, debug=debug), coords
    else:
        return pixel_to_coords_map_helper(coords, to_int=to_int, debug=debug)


def pixel_to_coords_map_helper(coords, to_int=False, debug=False):
    '''
    Helper function for pixel_to_coords_map
    Given a coords dictionary as returned by get_coordinates_from_text,
    returns a map from a pixel location to a [lon, lat] pair
    '''
    lon, lat = get_coordinate_span_directly(coords, locations=True)

    if any_coord_equal(lat[0], lat[1], 'LAT') or any_coord_equal(lon[0], lon[1], 'LON'):
        # raise RuntimeError('Number of valid coordinates found is not enough to extrapolate a mapping from pixels to coordinates')
        if debug:
            print('Number of valid coordinates found is not enough to extrapolate a mapping from pixels to coordinates')
        return None

    def to_latitude(y):
        pos1, v1, d1 = lat[0]
        pos2, v2, d2 = lat[1]
        v1, v2 = coord_to_abs(v1, d1), coord_to_abs(v2, d2)
        y1, y2 = pos1[1], pos2[1]
        slope = abs(float(v2 - v1) / (y2 - y1))
        lat_y = coord_from_abs(v1 - slope * (y - y1), 'LAT', to_int=to_int)
        return lat_y

    def to_longitude(x):
        pos1, v1, d1 = lon[0]
        pos2, v2, d2 = lon[1]
        east_to_west = lon[0][2] == 'E' and lon[1][2] == 'W'
        v1, v2 = coord_to_abs(v1, d1, flip_lon=east_to_west), coord_to_abs(v2, d2, flip_lon=east_to_west)
        x1, x2 = pos1[0], pos2[0]
        delta_v = 360 - v2 + v1 if east_to_west else v2 - v1
        slope = abs(float(delta_v) / (x2 - x1))
        lon_x = 0
        if east_to_west:
            lon_x = coord_from_abs(v2 - slope * (x - x2), 'LONG', flip_lon=east_to_west, to_int=to_int)
        else:
            lon_x = coord_from_abs(v1 + slope * (x - x1), 'LONG', flip_lon=east_to_west, to_int=to_int)
        return lon_x

    def pixel_to_coords(x, y=None):
        '''Given a pixel coord, returns the corresponding lon and lat pairs'''
        try:
            x.__iter__
        except (SyntaxError, SyntaxWarning, AttributeError) as e:
            return to_longitude(x), to_latitude(y)
        else:
            return to_longitude(x[0]), to_latitude(x[1])

    return pixel_to_coords


def pixel_to_coords_map_multiple_tries(img, path_given=False, num_iter=2, to_int=False, return_all_text=False, debug=False):
    '''
    Sequentially tries different padding and dist_threshold values to
    extract pixel_to_coords map, and returns the first valid map found
    This exists because different maps have different spacing/format of text
    and one of the current two ways usually works for most images till now
    If return_all_text is True, returns all coordinates extracted as well,
    inside a tuple of (pixel_to_coords function, coords)
    '''
    cv_img = cv2.imread(img) if path_given else np.array(img)
    paddings = [20, 10]
    dist_thresholds = [12, 12]
    n = min(len(paddings), num_iter)
    pixel_to_coords, coords = None, None
    for i in range(n):
        p, d = paddings[i], dist_thresholds[i]
        if debug:
            print('Trying with padding %d and thresh %d' % (p, d))
        pixel_to_coords = pixel_to_coords_map(cv_img, path_given=False, padding=p, dist_threshold=d, return_all_text=return_all_text, debug=debug)
        if return_all_text:
            pixel_to_coords, coords = pixel_to_coords

        if not pixel_to_coords:
            if debug:
                print('Failed with this padding and thresh')
            continue
        elif not valid_pixel_map(pixel_to_coords, cv_img.shape):
            if debug:
                print('Invalid map with this padding and thresh')
            continue
        else:
            break

    return (pixel_to_coords, coords) if return_all_text else pixel_to_coords
