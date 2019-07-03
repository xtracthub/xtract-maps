from location_extraction import load_city_index, load_border_index, extract_location_metadata

''' 
    
    This script handles the run-logic for the map-wizard Skluma module. The Docker container stored as a DockerHub Image
    should store all dependency downloads. Some 'difficult' ones include the shapely and opencv (cv2) downloads, 
    Google OCR Tesseract, and libgeos-dev (C binding for shapely). 
    
    This Docker Image should always run Python3.  

    @ScriptAuthor: Tyler J. Skluzacek (skluzacek@uchicago.edu) --- DockerImage/Workflow point of contact
    @ModuleAuthor: Rohan Kumar --- Module point of contact
    
'''


def get_indices():
    # Load city and border indices into memory.
    city_index = load_city_index()
    border_index = load_border_index()

    return (city_index, border_index)


# # Load file from shared volume.
# input_file = "fix_this.png"
#
# # Extract map-based metadata:
# metadata = extract_location_metadata(input_file, border_index, city_index, path_given=True, debug=False)

#Sample usage -- set debug == False to remove verbose text.
def extract_map_metadata(filename):

    indices = get_indices()
    city_index = indices[0]
    border_index = indices[1]
    metadata = extract_location_metadata(filename, border_index, city_index,path_given=True, debug=False)

    print(metadata)
    return(metadata)

import time

t0 = time.time()
extract_map_metadata('testpic.png')
t1 = time.time()
print(t1-t0)

""" Tyler and Rohan's 'messing around' code from 08/16/2017""" #TODO: Move logic to test-suite.
# from contouring import isolate_text_boxes
# import cv2
# from PIL import Image
#
# boxes = isolate_text_boxes(cv2.imread(mappy), return_contours=True)
# i = cv2.drawContours(cv2.imread(mappy), boxes, -1, (255,0,0), 2)
# Image.fromarray(i).show()
