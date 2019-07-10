import numpy as np
import json
from collections import OrderedDict
import cv2
from shapely.geometry import shape
import logging
from .contouring import valid_image
from .border_extraction import extract_borders, extract_borders_with_coordinates, borders_to_coordinates
from .coordinate_extraction import pixel_to_coords_map_multiple_tries, pixel_to_coords_to_span


# Setup for the debug logger
logging.basicConfig(format='%(asctime)s - %(filename)s - %(funcName)s - %('
                           'message)s', level=logging.DEBUG,
                    filename='debug.txt', filemode='w')
logger = logging.getLogger('logger')


# Attributes available in the country border index
attribute_names = ['NAME', 'NAME_LONG', 'TYPE', 'SOVEREIGNT', 'REGION_WB',
                   'SUBREGION', 'REGION_UN', 'CONTINENT', 'ECONOMY']

# Functions to Build / Load Country Border Index


def generate_country_coordinates_index(list_of_shapes):
    """"Takes a shapefile and returns a dictionary of attributes.

    Parameter:
    list_of_shapes (list): List of shapes from a shapefile.
    Currently supported format is the one found at:
    http://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-0-details/
    Coordinate format is the standard GCS_WGS_1984.

    Return:
    index (OrderedDict): OrderedDict with all attributes in
    attribute_names, a count id, and geometry exactly copied from the
    shapefile (including coordinates).
    """
    index = OrderedDict()
    for x in list_of_shapes:
        name = x['properties']['NAME_LONG']
        for a in attribute_names:
            attributes = OrderedDict({a: x['properties'][a]})
        item = OrderedDict()
        item['id'] = int(x['id'])
        item['attributes'] = attributes
        item['geometry'] = x['geometry']
        index[name] = item
    return index


def build_index_from_shapefile(shape_file_path, json_file_path):
    """Generates a country index from shapefile and dumps it to a json
    file.

    Parameters:
    shape_file_path (str): File path to shapefile format supported by
    generate_country_coordinates_index.
    json_file_path (str): File path to json file to dump country indexes
    to.

    Return:
    index (OrderedDict): OrderedDict with all attributes in
    attribute_names, a count id, and geometry exactly copied from the
    shapefile (including coordinates).
    """
    fi = open(shape_file_path, 'r')
    list_of_shapes = list(fi)
    fi.close()
    index = generate_country_coordinates_index(list_of_shapes)
    with open(json_file_path, 'w') as f:
        json.dump(index, f)
    return index


def load_border_index(index_name='unit'):
    """Loads a country-coordinate index from a json file.

    Parameter:
    index_name (str): The name of the index you are loading data from.
    ('country', 'unit', 'subunit', 'sovereign')

    Help on choosing index_name for your use:
    country: Each country's dependent overseas regions are the same
    entity.
    unit: Dependent overseas regions of a country are separate entities.
    subunit: Countries subdivided by non-contiguous units, for example,
    Alaska and the rest of the US are separate entities.
    sovereign: Does not distinguish b/w metropolitan and
    semi-independent portions of a state or its constituent countries.

    Return:
    (OrderedDict): Ordered dictionary of information from index.
    """
    index_names = ['country', 'unit', 'subunit', 'sovereign']

    if index_name.lower() not in index_names:
        raise ValueError('index_name must be one of %s' % index_names)

    name_to_index = {
        'country': 'xtract_maps/country_shapefiles/country.json',
        'unit': 'xtract_maps/country_shapefiles/map_units.json',
        'subunit': 'xtract_maps/country_shapefiles/map_subunits.json',
        'sovereign': 'xtract_maps/country_shapefiles/map_sovereign.json',
    }

    with open(name_to_index[index_name]) as f:
        index = json.load(f)
    return OrderedDict(index)


def load_city_index(path='xtract_maps/city_index.json'):
    """Loads a city index from a JSON file.

    Parameter:
    path (str): Path to city index you are trying to load. Must be in the
    format of the database from
    https://www.maxmind.com/en/free-world-cities-database.

    Return:
    (list): Returns a list of dictionaries (one per city).
    """
    with open(path, 'r') as f:
        return json.load(f)

# Functions to extract location metadata


def contains(big, small, ignore_case=False):
    """Checks whether the words in small are a subsequence of those in
    big.

    Parameters:
    big (str): String that small may be a subsequence of.
    small (str): String that might be a subsequence of big.
    ignore_case (bool): Whether to ignore uppercases and lowercases.
    """
    if not small:
        return False
    if ignore_case:
        big, small = big.lower(), small.lower()
    big_parts, small_parts = big.split(), small.split()
    big_n, small_n = len(big_parts), len(small_parts)
    for i in range(big_n - small_n + 1):
        for j in range(small_n):
            if not big_parts[i+j] == small_parts[j]:
                break
        else:
            return True
    return False


def get_location_names(test_locations, cities_index):
    """Returns a list of (location, type) tuples from cities_index found
    in test_locations.

    test_locations (iter): Iterable of strings to check.
    cities_index (list): List of dictionaries as returned by
    load_city_index.

    Return:
    (list(tuple)): List of 2-tuple containing the location and the
    location type.
    """
    # key names available: ['city', 'city_ascii', 'province', 'iso2',
    # 'iso3', 'country', 'lat', 'lng', 'pop']
    given_text = ' '.join(test_locations)
    locations = list()
    for item in cities_index:
        for key in ['country', 'province', 'city']:
            if contains(given_text, item[key], ignore_case=True):
                locations.append((item[key], key))
        if contains(given_text, item['iso3'], ignore_case=False):
            locations.append((item['iso3'], 'iso3'))

    return list(set(locations))


def overlapping_shapes(border, index):
    """Returns a list of (country, data) tuples for all countries in the
    index that overlap with the given border.

    Parameter:
    border (list): List of (lon, lat) points representing a
    polygon/border.
    index (OrderedDict): Ordered dictionary as returned by
    load_border_index.

    Return:
    (list): List of (country, data) tuples that have an intersection
    with border.
    """
    given_shape = shape({'coordinates': [border], 'type': 'Polygon'})

    return [(country, data) for (country, data) in index.items()
            if given_shape.intersects(shape(data['geometry']))]


def overlapping_shapes_data(border, index, attributes=['NAME_LONG'],
                            unique=True):
    """Maps each valid attribute to a list of values for that attribute
    found in overlapping_shapes(border, index).

    Parameters:
    border (list): List of (lon, lat) points representing a
    polygon/border.
    index (OrderedDict): Ordered dictionary as returned by
    load_border_index.
    attributes (list(str)): List of attributes from attribute_names to
    be returned.
    unique (bool): If True, returns unique values found for each
    attribute.

    Return:
    result (OrderedDict): Ordered dictionary of attributes of locations
    from border.
    """
    attributes = [
            x for x in map(lambda x: x.upper(), attributes)
            if x in attribute_names
        ]
    shapes = overlapping_shapes(border, index)
    result = OrderedDict()
    for attr in attributes:
        result[attr] = list()
        for _, data in shapes:
            result[attr] += [data['attributes'][attr]]
        if unique:
            result[attr] = list(set(result[attr]))
    return result


def borders_to_regions(borders, border_index, attributes=['NAME_LONG'],
                       unique=True):
    """Gets the overlapping_shapes_data result for each border in
    borders.

    Parameters:
    borders (list): List of (lon, lat) points representing a
    polygon/border.
    borders_index (OrderedDict): Ordered dictionary as returned by
    load_border_index.
    attributes (list(str)): List of attributes from attribute_names to
    be returned.
    unique (bool): If True, returns unique values found for each
    attribute.

    Return:
    results (list(OrderedDict)): List of attributes for each border in
    borders.
    """
    results = [
        overlapping_shapes_data(border, border_index, attributes=attributes,
                                unique=unique)
        for border in borders
    ]
    return results


def extract_regions(img, border_index, path_given=False, approximation=0.01,
                    attributes=['NAME_LONG'], unique=True):
    """Given an image of a map, extracts borders present in it using the
    coordinate_extraction module, and then uses the border_index given
    to return the regions which overlap with these borders.

    Parameters:
    img (OpenCV image or file path): OpenCV image or file path to
    extract regions from.
    borders_index (OrderedDict): Ordered dictionary as returned by
    load_border_index.
    path_given (bool): Whether a file path is given in img.
    attributes (list(str)): List of attributes from attribute_names to
    be returned.
    approximation (float): Error distance between approximation and
    original contour.
    unique (bool): If True, returns unique values found for each
    attribute.

    Return:
    results (list(OrderedDict)): List of attributes for each region in
    img.
    """
    borders = extract_borders_with_coordinates(img=img, path_given=path_given,
                                               approximation=approximation,
                                               absolute=True)

    return borders_to_regions(borders, border_index, attributes=attributes,
                              unique=unique)


def extract_location_metadata_with_borders(img, border_index, city_index,
                                           path_given=False,
                                           approximation=0.01,
                                           attributes=['NAME_LONG'],
                                           unique=True, debug=False):
    """Extracts coordinate-span, regions-list, and directly-extract-text
    from a map image.

    Using the coordinate_extraction module, extracts the coordinate span
    of the given map. Then using the border_extraction module, gets
    coordinate-borders of objects in the map, feeding them into
    borders_to_regions to get regions that the map covers. Also extracts
    location names from the text extracted from the image.
    Returns a (coordinate-span, regions-list, directly-extracted-text)
    tuple with values as returned by the corresponding functions.

    Parameters:
    img (OpenCV image or file path): OpenCV image or file path to
    extract metadata from.
    borders_index (OrderedDict): Ordered dictionary as returned by
    load_border_index.
    city_index(list): List of dictionaries as returned by
    load_city_index.
    path_given (bool): Whether a file path is given in img.
    approximation (float): Error distance between approximation and
    original contour.
    attributes (list(str)): List of attributes from attribute_names to
    be returned.
    unique (bool): If True, returns unique values found for each
    attribute.

    Return:
    (tuple): 3-tuple of coordinate span of img, regions found in img,
    and text directly found in img.
    """
    cv_img = cv2.imread(img) if path_given else np.array(img)
    img_dim = cv_img.shape

    # Extract pixel-coordinate map
    pixel_to_coords, coords = pixel_to_coords_map_multiple_tries(cv_img,
                                                                 path_given=False,
                                                                 return_all_text=True,
                                                                 debug=debug)
    direct_text = get_location_names(coords['other'].values(), city_index)

    if not pixel_to_coords:     # valid pixel_to_coords not found
        return None, None, direct_text

    coord_span = pixel_to_coords_to_span(pixel_to_coords, img_dim)

    # Extract borders, and then regions found
    pixel_borders = extract_borders(cv_img, path_given=False,
                                    approximation=approximation, close=True,
                                    debug=debug)
    coord_borders = borders_to_coordinates(pixel_borders, pixel_to_coords,
                                           absolute=True)
    regions = borders_to_regions(coord_borders, border_index,
                                 attributes=attributes, unique=unique)

    return coord_span, regions, direct_text


def extract_location_metadata_without_borders(img, border_index, city_index,
                                              path_given=False, edge=20,
                                              attributes=['NAME_LONG'],
                                              unique=True, debug=False):
    """Extracts coordinate-span, regions-list, and directly-extract-text
    from a map image.

    Using the coordinate_extraction module, extracts the coordinate span
    of the given map. Feeds this rectangular span directly into the
    border_index to search for overlapping countries. Note that this
    loses the specificity of border-to-region mappings even though it
    gives the same regions. However, since it skips the border
    extraction step and avoids repeated searches across the
    border_index, it's faster.

    Parameters:
    img (OpenCV image or file path): OpenCV image or file path to
    extract metadata from.
    borders_index (OrderedDict): Ordered dictionary as returned by
    load_border_index.
    city_index(list): List of dictionaries as returned by
    load_city_index.
    path_given (bool): Whether a file path is given in img.
    edge (int): Number of pixels from image border to consider extent
    of map.
    attributes (list(str)): List of attributes from attribute_names to
    be returned.
    unique (bool): If True, returns unique values found for each
    attribute.

    Return:
    (tuple): 3-tuple of coordinate span of img, regions found in img,
    and text directly found in img.
    """
    cv_img = cv2.imread(img) if path_given else np.array(img)
    img_dim = cv_img.shape

    # extract pixel-coordinate map
    pixel_to_coords, coords = pixel_to_coords_map_multiple_tries(cv_img,
                                                                 path_given=False,
                                                                 return_all_text=True,
                                                                 debug=debug)
    direct_text = get_location_names(coords['other'].values(), city_index)

    if not pixel_to_coords:     # valid pixel_to_coords not found
        return None, None, direct_text

    coord_span = pixel_to_coords_to_span(pixel_to_coords, img_dim, edge=edge)
    lon, lat = pixel_to_coords_to_span(pixel_to_coords, img_dim, edge=edge,
                                       absolute=True)
    span_box = np.array([
        (lon[1], lat[0]),
        (lon[0], lat[0]),
        (lon[0], lat[1]),
        (lon[1], lat[1]),
    ])
    regions = borders_to_regions([span_box], border_index,
                                 attributes=attributes, unique=unique)

    return coord_span, regions, direct_text


def extract_location_metadata(img, border_index, city_index, use_borders=True,
                              path_given=False, approximation=0.01, edge=20,
                              attributes=['NAME_LONG'], unique=True,
                              debug=False):
    """Extracts coordinate-span, regions-list, and directly-extract-text
    from a map image.

    Parameters:
    img (OpenCV image or file path): OpenCV image or file path to
    extract metadata from.
    borders_index (OrderedDict): Ordered dictionary as returned by
    load_border_index.
    city_index(list): List of dictionaries as returned by
    load_city_index.
    use_borders (bool): Whether to extract metadata using borders or
    without.
    path_given (bool): Whether a file path is given in img.
    approximation (float): Error distance between approximation and
    original contour.
    edge (int): Number of pixels from image border to consider extent
    of map.
    attributes (list(str)): List of attributes from attribute_names to
    be returned.
    unique (bool): If True, returns unique values found for each
    attribute.

    Return:
    (tuple): 3-tuple of coordinate span of img, regions found in img,
    and text directly found in img.
    """
    if not valid_image(img):
        if debug:
            logger.debug('Image not valid')
        return None, None, None
    elif use_borders:
        return extract_location_metadata_with_borders(img, border_index,
                                                      city_index,
                                                      path_given=path_given,
                                                      approximation=approximation,
                                                      attributes=attributes,
                                                      unique=unique,
                                                      debug=debug)
    else:
        return extract_location_metadata_without_borders(img, border_index,
                                                         city_index,
                                                         path_given=path_given,
                                                         edge=edge,
                                                         attributes=attributes,
                                                         unique=unique,
                                                         debug=debug)

# Test code, please ignore

# def main(use_borders=True):
#     images = [
#         # '../../Image_Processing/pub8_images/CAIBOX_2009_map.jpg',
#         # '../../Image_Processing/pub8_images/GOMECC2_map.jpg',
#         # '../../Image_Processing/pub8_images/EQNX_2015_map.jpg',
#         # '../../Image_Processing/pub8_images/Marion_Dufresne_map_1991_1993.jpg',
#         # '../../Image_Processing/pub8_images/P16S_2014_map.jpg',
#         # '../../Image_Processing/pub8_images/Oscar_Dyson_map.jpg',
#         # '../../Image_Processing/pub8_images/Bigelow2015_map.jpg',
#         # '../../Image_Processing/pub8_images/A16S_2013_map.jpg',
#         # '../../Image_Processing/pub8_images/woce_a25.gif',
#         # '../../Image_Processing/pub8_images_2/pub8.oceans.save.SAVE.jpg',
#         '../../Image_Processing/pub8_images/Map_WEST_Coast_Cruise_2007.jpg',
#         # '../../Image_Processing/us-states.png',
#         'doesnotexist.blah',
#         ]
#
#     # input('Press enter to run')
#     index = load_border_index()
#     city_index = load_city_index()
#     for image in images:
#         print('For %s:' % image)
#         span, regions, direct_regions = extract_location_metadata(image, index, city_index, path_given=True, use_borders=use_borders, debug=False)
#         print(span, '\n')
#         print(regions, '\n')
#         print(direct_regions, '\n')
#
#
# if __name__ == '__main__':
#     main(use_borders=True)
