FROM python:latest

MAINTAINER Ryan Wong

RUN apt-get update

# Copy files
COPY unionfind.py contouring.py text_extraction.py coordinate_extraction.py border_extraction.py location_extraction.py city_index.json xtract_maps_main.py /

COPY country_shapefiles /country_shapefiles

# Install dependencies
RUN pip install numpy opencv-python Pillow shapely pytesseract
RUN apt install tesseract-ocr -y libtesseract-dev -y

ENTRYPOINT ["python", "xtract_maps_main.py"]
