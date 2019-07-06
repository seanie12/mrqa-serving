# This image is basically for MRQA 2019 submission
FROM nvidia/cuda:9.0-cudnn7-runtime
MAINTAINER Jangwon Park "adieujw@gmail.com"


############################################################
# Common steps (must be the same in the CPU and GPU images)
RUN apt-get update && apt-get install -y \
    apt-transport-https \
    iputils-ping \
    git \
    python2.7 \
    python-pip \
    python-dev \
    python-software-properties \
    python-tk \
    software-properties-common \
    build-essential \
    cmake \
    libhdf5-dev \
    swig \
    wget \
    curl

## Python 3.6
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update -y  && \
    apt-get install python3.6 -y \
    python3.6-venv \
    python3.6-dev \
    python3-software-properties

RUN curl https://bootstrap.pypa.io/get-pip.py | python3.6

### Without this Python thinks we're ASCII and unicode chars fail
ENV LANG C.UTF-8

## Python 3 packages
RUN pip3 install -U pip

############################################################
# GPU-specific commands
RUN pip3 install -U \
    torch \
    torchvision

RUN pip3 install pytorch-pretrained-bert flask json-lines
