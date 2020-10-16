FROM nvcr.io/nvidia/tensorflow:20.09-tf1-py3 AS nvidia-base

WORKDIR /app

RUN adduser --disabled-password --gecos '' fednuser && \
    adduser fednuser sudo && \
    echo '%sudo ALL=(ALL:ALL) ALL' >> /etc/sudoers && \
    chown -R fednuser /app && \
    rm -rf /var/lib/apt/lists/*

FROM nvidia-base AS python-modules

COPY fedn_requirements.txt fedn_requirements.txt
COPY fedbird_requirements.txt fedbird_requirements.txt

RUN TF_ENABLE_DEPRECATION_WARNINGS=1 python3 -c 'import tensorflow as tf; print(tf.__version__)' && \
    pip install --use-feature=2020-resolver -r fedn_requirements.txt -r fedbird_requirements.txt 

FROM python-modules AS scaleout-fedn

RUN git clone -b v0.1.0 https://github.com/scaleoutsystems/fedn.git && \
    pip install --use-feature=2020-resolver -e fedn/fedn

FROM scaleout-fedn AS fedbird

RUN mkdir certs
