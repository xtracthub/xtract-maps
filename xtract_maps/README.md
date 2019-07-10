# Xtractor Map Location

This pipeline consists of 5 main modules, with each building directly
on the one before it. There is also a 6th UnionFind utility module.

The hierarchy of modules is as follows:

### contouring.py
        Employs OpenCV's contouring algorithms and the UnionFind utility
        module to find regions of an image which may contain text.

### text_extraction.py
        Uses the contouring module to extract text boxes from the image
        and then feeds them into Tesseract to extract text.

### coordinate_extraction.py
        Post-processes the text received from the text_extraction module
        to get coordinates (longitude and latitude) of the map, including
        a pixel-to-coordinates map for the image.

### border_extraction.py
        Using OpenCV's contouring algorithms, finds borders of regions in
        the map, for example, land masses. Also feeds these borders into
        the pixel-to-coordinates map from the coordinate_extraction module
        to get (longitude, latitude) borders of the regions found.

### location_extraction.py
        Uses the (longitude, latitude) borders from the border_extraction
        module to search across a previously stored index of country borders
        to extract which regions of the world are present in the map. Includes
        option to go directly from coordinate span to regions. Also extracts
        any available locations directly from text found in the map.


Note: Contact Rohan Kumar for any clarifications needed
