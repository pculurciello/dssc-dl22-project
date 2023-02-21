FROM nvidia/cuda:12.0.1-base-ubuntu20.04

# Set bash as the default shell
ENV SHELL=/bin/bash

# Create a working directory
WORKDIR /app/

# Build with some basic utilities
RUN apt-get update && apt-get install -y \
    python3-pip apt-utils vim git locate cargo

RUN DEBIAN_FRONTEND='noninteractive' apt-get install -y --no-install-recommends \
    libsndfile1 libsndfile1-dev libsndio-dev
    
# alias python='python3'
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip install \
    jupyter-events==0.5.0 \
    numpy==1.23.5 \
    torchvision==0.9.1 \
    fastaudio \
    jupyterlab 

COPY DSSC_DL22_project .

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
EXPOSE 8888
