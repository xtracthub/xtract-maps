FROM python:3.6

MAINTAINER Ryan Wong

# Copy files
COPY unionfind.py contouring.py text_extraction.py coordinate_extraction.py border_extraction.py location_extraction.py city_index.json xtract_maps_main.py /

COPY country_shapefiles /country_shapefiles

# Install dependencies
RUN pip install numpy opencv-python Pillow shapely pytesseract git+https://github.com/Parsl/parsl git+https://github.com/DLHub-Argonne/home_run
RUN apt install tesseract-ocr -y libtesseract-dev -y

#ENTRYPOINT ["python", "xtract_maps_main.py"]
