FROM python:latest

MAINTAINER Ryan Wong

RUN apt-get update

# Copy files
COPY unionfind.py /
COPY contouring.py /
COPY text_extraction.py /
COPY coordinate_extraction.py /
COPY border_extraction.py /
COPY location_extraction.py /
COPY xtract_maps_main.py /

# Install dependencies
RUN pip install numpy opencv-python Pillow shapely
RUN apt install tesseract-ocr -y
RUN apt install libtesseract-dev -y
RUN pip install pytesseract

CMD ["python", "xtract_maps_main.py"]
