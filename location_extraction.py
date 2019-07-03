import numpy as np
import json
from collections import OrderedDict

import cv2
from shapely.geometry import shape, mapping, Point, Polygon, MultiPolygon
from shapely.strtree import STRtree

from contouring import valid_image
from border_extraction import extract_borders, extract_borders_with_coordinates, borders_to_coordinates
from coordinate_extraction import pixel_to_coords_map, pixel_to_coords_map_multiple_tries, pixel_to_coords_to_span


# attributes available in the country border index
attribute_names = ['NAME', 'NAME_LONG', 'TYPE', 'SOVEREIGNT', 'REGION_WB', 'SUBREGION', 'REGION_UN', 'CONTINENT', 'ECONOMY']

# -----------------------------------------------------------------------------
#                Functions to Build / Load Country Border Index
# -----------------------------------------------------------------------------

def generate_country_coordinates_index(list_of_shapes):
    '''
    :param list_of_shapes: list of shapes from a shapefile
    Currently supported format is the one found at:
    http://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-0-details/
    Coordinate format is the standard GCS_WGS_1984
    The index returned is a (country long name)-to-data OrderedDict
    Data includes all attributes in attribute_names, a count id, and
    geometry exactly copied from the shapefile (which includes coordinates)
    '''
    index = OrderedDict()
    for x in list_of_shapes:
        name = x['properties']['NAME_LONG']
        attributes = OrderedDict({a: x['properties'][a] for a in attribute_names})
        item = OrderedDict()
        item['id'] = int(x['id'])
        item['attributes'] = attributes
        item['geometry'] = x['geometry']
        index[name] = item
    return index


def build_index_from_shapefile(shape_file_path, json_file_path):
    '''
    Given the path to a shapefile in the format supported by
    generate_country_coordinates_index, generates the country index
    and dumps it to a json file
    '''
    fi = open(shape_file_path, 'r')
    l = list(fi)
    close()
    index = generate_country_coordinates_index(l)
    with open(json_file_path, 'w') as f:
        json.dump(index, f)
    return index


def load_border_index(index_name='unit'):
    '''
    Loads a country-coordinate index from a json file

    Help on choosing index_name for your use:
    :country: each country's dependent overseas regions are the same entity
    :unit: dependent overseas regions of a country are separate entities
    :subunit: countries subdivided by non-contiguous units, for example,
    Alaska and the rest of the US are separate entities
    :sovereign: does not distinguish b/w metropolitan and semi-independent
    portions of a state or its constituent countries
    '''
    index_names = ['country', 'unit', 'subunit', 'sovereign']

    if index_name.lower() not in index_names:
        raise ValueError('index_name must be one of %s' % index_names)
        return None

    name_to_index = {
        'country': 'country_shapefiles/country.json',
        'unit': 'country_shapefiles/map_units.json',
        'subunit': 'country_shapefiles/map_subunits.json',
        'sovereign': 'country_shapefiles/map_sovereign.json',
    }

    index = None
    with open(name_to_index[index_name]) as f:
        index = json.load(f)
    return OrderedDict(index)


def load_city_index(path='city_index.json'):
    '''
    Loads a city index from a JSON file, in the format of the database from
    https://www.maxmind.com/en/free-world-cities-database
    If correct file given, returns a list of dictionaries (one dict per city)
    '''
    with open(path, 'r') as f:
        return json.load(f)


# -----------------------------------------------------------------------------
#                   Functions to Extract Location Metadata
# -----------------------------------------------------------------------------

def remove_error_borders(borders):
    '''
    Does not work as needed, please ignore this function
    Old function to try and correct problems that occur with borders that
    cross the international date line (east to west messes polygon shapes)
    '''
    new_borders = list()
    for border in borders:
        new_border = border
        e_count, w_count = 0, 0
        for (x,y) in border:
            if x >= 90:
                e_count += 1
            elif x <= -90:
                w_count += 1
        if e_count and w_count and e_count > w_count:
            new_border = [
                (x+360, y) if x < 0 else (x, y)
                for (x, y) in border
            ]
        new_borders.append(new_border)
    return new_borders


def contains(big, small, ignore_case=False):
    '''Checks whether the words in small are a subsequence of those in big'''
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
    '''
    Returns a list of (location, type) tuples from city_index found in
    test_locations
    :param test_locations: iterable of strings to check
    :param city_index: list of dictionaries, in the format of the cities
    database from https://www.maxmind.com/en/free-world-cities-database,
    as returned by load_city_index
    '''
    # key names available: ['city', 'city_ascii', 'province', 'iso2', 'iso3', 'country', 'lat', 'lng', 'pop']
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
    '''
    :param border: list of (lon, lat) points representing a polygon/border
    :param index: country-coordinate index as returned by load_border_index
    Returns a list of (country, data) pairs for all countries in the
    index which have any overlap with the given border
    '''
    given_shape = shape({'coordinates': [border], 'type': 'Polygon'})

    return [(country, data) for (country, data) in index.items() if given_shape.intersects(shape(data['geometry']))]


def overlapping_shapes_data(border, index, attributes=['NAME_LONG'], unique=True):
    '''
    :param border: list of (lon, lat) points representing a polygon/border
    :param index: country-coordinate index as returned by load_border_index
    :param attributes: which attributes out of attribute_names to be returned
    :param unique: if True, returns unique values found for each attribute
    Returns an OrderedDict mapping each valid attribute to a list of values
    for that attribute found in overlapping_shapes(border, index)
    '''
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


def borders_to_regions(borders, border_index, attributes=['NAME_LONG'], unique=True):
    '''Gets the overlapping_shapes_data result for each border in borders'''
    # convert to (lon, lat) format (not needed anymore because I switched)
    # borders = [[(x,y) for (y,x) in border] for border in borders]
    results = [
        overlapping_shapes_data(border, border_index, attributes=attributes, unique=unique)
        for border in borders
    ]
    return results


def extract_regions(img, border_index, path_given=False, approximation=0.01, attributes=['NAME_LONG'], unique=True):
    '''
    Given an image of a map, extracts borders present in it using the
    coordinate_extraction module, and then uses the border_index given
    to return the regions which overlap with these borders
    '''
    borders = extract_borders_with_coordinates(img=img, path_given=path_given, approximation=approximation, absolute=True)

    return borders_to_regions(borders, border_index, attributes=attributes, unique=unique)


def extract_location_metadata_with_borders(img, border_index, city_index, path_given=False, approximation=0.01, attributes=['NAME_LONG'], unique=True, debug=False):
    '''
    Using the coordinate_extraction module, extracts the coordinate span
    of the given map. Then using the border_extraction module, gets
    coordinate-borders of objects in the map, feeding them into
    borders_to_regions to get regions that the map covers. Also extracts
    location names from the text extracted from the image.
    Returns a (coordinate-span, regions-list, directly-extracted-text)
    tuple with values as returned by the corresponding functions
    '''
    cv_img = cv2.imread(img) if path_given else np.array(img)
    img_dim = cv_img.shape

    # extract pixel-coordinate map
    pixel_to_coords, coords = pixel_to_coords_map_multiple_tries(cv_img, path_given=False, return_all_text=True, num_iter=2, debug=debug)
    direct_text = get_location_names(coords['other'].values(), city_index)

    if not pixel_to_coords:     # valid pixel_to_coords not found
        return None, None, direct_text

    coord_span = pixel_to_coords_to_span(pixel_to_coords, img_dim)

    # extract borders, and then regions found
    pixel_borders = extract_borders(cv_img, path_given=False, approximation=approximation, close=True, close_window_size=8, debug=debug)
    coord_borders = borders_to_coordinates(pixel_borders, pixel_to_coords, absolute=True)
    regions = borders_to_regions(coord_borders, border_index, attributes=attributes, unique=unique)

    return coord_span, regions, direct_text


def extract_location_metadata_without_borders(img, border_index, city_index, path_given=False, edge=20, attributes=['NAME_LONG'], unique=True, debug=False):
    '''
    Using the coordinate_extraction module, extracts the coordinate span
    of the given map. Feeds this rectangular span directly into the
    border_index to search for overlapping countries. Note that this loses
    the specificity of border-to-region mappings even though it gives the same
    regions. However, since it skips the border extraction step and avoids
    repeated searches across the border_index, it's faster.
    '''
    cv_img = cv2.imread(img) if path_given else np.array(img)
    img_dim = cv_img.shape

    # extract pixel-coordinate map
    pixel_to_coords, coords = pixel_to_coords_map_multiple_tries(cv_img, path_given=False, return_all_text=True, num_iter=2, debug=debug)
    direct_text = get_location_names(coords['other'].values(), city_index)

    if not pixel_to_coords:     # valid pixel_to_coords not found
        return None, None, direct_text

    coord_span = pixel_to_coords_to_span(pixel_to_coords, img_dim, edge=edge)
    lon, lat = pixel_to_coords_to_span(pixel_to_coords, img_dim, edge=edge, absolute=True)
    span_box = np.array([
        (lon[1], lat[0]),
        (lon[0], lat[0]),
        (lon[0], lat[1]),
        (lon[1], lat[1]),
    ])
    regions = borders_to_regions([span_box], border_index, attributes=attributes, unique=unique)

    return coord_span, regions, direct_text


def extract_location_metadata(img, border_index, city_index, use_borders=True, path_given=False, approximation=0.01, edge=20, attributes=['NAME_LONG'], unique=True, debug=False):
    '''
    You will most likely want to use this main wrapper function. This lets
    you choose between the with and without border extraction options. Also,
    checks whether given image is valid for OpenCV or not before processing.

    :param img: OpenCV image if path_given else path to image of map
    :param border_index: for index of country borders, use load_border_index
    :param city_index: for index of city names, use load_city_index
    :param use_borders: whether to use border extraction or skip that step
    :param approximation: in case of with borders, a measure of how much borders should be approximated, no approximation if None
    :param edge: in case of without borders, number of pixels away from the
    edge of the image to consider the span of the map
    :param attributes: list of location attributes returned from the
    border_index, must be a subset of attribute_names defined at the top
    :param unique: whether returned attributes should be uniqued out or not

    Returns a (coordinate-span, regions-list, directly-extracted-text) tuple
    :coordinate-span: is as returned from get_coordinate_span
    :regions-list: is as returned from extract_regions
    :directly-extracted-text: is as returned from get_location_names
    '''
    if not valid_image(img):
        if debug:
            print('Image not valid')
        return None, None, None
    elif use_borders:
        return extract_location_metadata_with_borders(img, border_index, city_index, path_given=path_given, approximation=approximation, attributes=attributes, unique=unique, debug=debug)
    else:
        return extract_location_metadata_without_borders(img, border_index, city_index, path_given=path_given, edge=edge, attributes=attributes, unique=unique, debug=debug)


def main(use_borders=True):
    images = [
        # '../../Image_Processing/pub8_images/CAIBOX_2009_map.jpg',
        # '../../Image_Processing/pub8_images/GOMECC2_map.jpg',
        # '../../Image_Processing/pub8_images/EQNX_2015_map.jpg',
        # '../../Image_Processing/pub8_images/Marion_Dufresne_map_1991_1993.jpg',
        # '../../Image_Processing/pub8_images/P16S_2014_map.jpg',
        # '../../Image_Processing/pub8_images/Oscar_Dyson_map.jpg',
        # '../../Image_Processing/pub8_images/Bigelow2015_map.jpg',
        # '../../Image_Processing/pub8_images/A16S_2013_map.jpg',
        # '../../Image_Processing/pub8_images/woce_a25.gif',
        # '../../Image_Processing/pub8_images_2/pub8.oceans.save.SAVE.jpg',
        '../../Image_Processing/pub8_images/Map_WEST_Coast_Cruise_2007.jpg',
        # '../../Image_Processing/us-states.png',
        'doesnotexist.blah',
        ]

    # input('Press enter to run')
    index = load_border_index()
    city_index = load_city_index()
    for image in images:
        print('For %s:' % image)
        span, regions, direct_regions = extract_location_metadata(image, index, city_index, path_given=True, use_borders=use_borders, debug=False)
        print(span, '\n')
        print(regions, '\n')
        print(direct_regions, '\n')


if __name__ == '__main__':
    main(use_borders=True)
