FROM python:latest

MAINTAINER Ryan Wong

RUN apt-get update

# Copy files
COPY code code
COPY data data
COPY tests tests

# Install dependencies
RUN pip install numpy opencv-python Pillow shapely pytesseract
RUN apt install tesseract-ocr -y libtesseract-dev -y

ENTRYPOINT ["python", "code/xtract_maps_main.py"]
