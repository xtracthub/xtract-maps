from location_extraction import load_city_index, load_border_index, extract_location_metadata
import time
import argparse

"""This script handles the run-logic for the map-wizard Skluma module. 
The Docker container stored as a DockerHub Image should store all dependency 
downloads. Some 'difficult' ones include the shapely and opencv (cv2) 
downloads, Google OCR Tesseract, and libgeos-dev (C binding for shapely). 

This Docker Image should always run Python3.  
@ScriptAuthor: Tyler J. Skluzacek (skluzacek@uchicago.edu) --- DockerImage/Workflow point of contact
@ModuleAuthor: Rohan Kumar --- Module point of contact

"""


def get_indices():
    """Loads city and border indexes.
    Return:
    (tuple): 2-tuple of city index and border index.
    """
    city_index = load_city_index()
    border_index = load_border_index()

    return (city_index, border_index)


def extract_map_metadata(filename, debug=False):
    """Extracts map metadata from a map image.
    Parameter:
    filename (str): Path to map image.
    Return:
    metadata (tuple): 3-tuple of coordinate span of img, regions found in img,
    and text directly found in img.
    """
    t0 = time.time()
    indices = get_indices()
    city_index = indices[0]
    border_index = indices[1]
    metadata = {"map": extract_location_metadata(filename, border_index,
                                                 city_index, path_given=True,
                                                 debug=debug)}
    metadata.update({"extract time": time.time() - t0})

    return metadata


if __name__ == "__main__":
    """Takes file paths from command line and returns metadata.
    Arguments:
    --path (File path): File path of map image file.
    --debug (bool): Whether to turn on debug mode.
    Returns:
    meta (insert type here): 3-tuple of coordinate span of img, regions found 
    in img, and text directly found in img.
    t1 - t0 (float): Time it took to retrieve map metadata.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='File system path to file.',
                        required=True)
    parser.add_argument('--debug', help='Whether to turn on debug mode.',
                        required=False, default=False)

    args = parser.parse_args()

    meta = extract_map_metadata(args.path, args.debug)
    print(meta)
