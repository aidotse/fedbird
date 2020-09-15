FROM nvcr.io/nvidia/tensorflow:20.08-tf2-py3 AS nvidia-base

WORKDIR /app

FROM nvidia-base AS python-modules

RUN python3 -c 'import tensorflow as tf; print(tf.__version__)' && \
    pip install pandas scikit-learn keras

FROM python-modules AS scaleout-fedn

RUN git clone https://github.com/scaleoutsystems/fedn.git && \
    pip install -e fedn/cli && \
    pip install -e fedn/sdk

FROM scaleout-fedn AS fed-bird-client

COPY src /app/client/
COPY project.yaml /app/client/
COPY data /app/data
