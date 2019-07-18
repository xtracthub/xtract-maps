#!/bin/bash

IMAGE_NAME='xtract_maps_image'

docker rmi -f $IMAGE_NAME

docker build -t $IMAGE_NAME .
