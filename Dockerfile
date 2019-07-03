# TODO: Sample Dockerfile. Change to fit Rohan's Spec.

FROM python:2.7-alpine
RUN apk add --no-cache g++ && \
    ln -s /usr/include/locale.h /usr/include/xlocale.h && \
    pip install numpy==1.12.0 && \
    pip install pandas==0.19.2


# Update
RUN apk add --update python py-pip

# Install app dependencies
RUN pip install globus_sdk

# Add emergency Docker-volume.
VOLUME ["/DataVolume1"]

# Bundle app source
COPY /pysrc/avsc_writer.py /src/avsc_writer.py
COPY /pysrc/close_process.py /src/close_process.py
COPY /pysrc/docker_mount.py /src/docker_mount.py
COPY /pysrc/csv_parser.py /src/csv_parser.py
COPY /pysrc/globus_connect.py /src/globus_connect.py
COPY /pysrc/Initialize.py /src/Initialize.py
COPY /pysrc/Main.py /src/Main.py
COPY /pysrc/metadata_util.py /src/metadata_util.py
COPY /pysrc/petrel_scanner.py /src/petrel_scanner.py
COPY /pysrc/pub8_list.txt /src/pub8_list.txt
COPY /pysrc/test_file.txt /src/test_file.txt
COPY /pysrc/type_inference.py /src/type_inference.py

#CMD ["python", "/src/Main.py"]
CMD ["python", "/src/docker_mount.py"]