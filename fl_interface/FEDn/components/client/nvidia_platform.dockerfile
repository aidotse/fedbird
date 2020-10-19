FROM nvcr.io/nvidia/tensorflow:20.08-tf1-py3 AS nvidia-base

WORKDIR /app

FROM nvidia-base AS python-modules

COPY requirements.txt requirements.txt
RUN python3 -c 'import tensorflow as tf; print(tf.__version__)' && \
    pip install --use-feature=2020-resolver -r requirements.txt 

FROM python-modules AS scaleout-fedn

RUN git clone -b v0.1.0 https://github.com/aidotse/fedn.git && \
    pip install --use-feature=2020-resolver -e fedn/fedn

FROM scaleout-fedn AS fedn-bird-client

RUN mkdir -p certs
