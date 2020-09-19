FROM nvcr.io/nvidia/tensorflow:20.08-tf2-py3 AS nvidia-base

WORKDIR /app

FROM nvidia-base AS python-modules

COPY requirements.txt requirements.txt
RUN python3 -c 'import tensorflow as tf; print(tf.__version__)' && \
    pip install --use-feature=2020-resolver -r requirements.txt 

FROM python-modules AS scaleout-fedn

RUN git clone https://github.com/scaleoutsystems/fedn.git && \
    pip install --use-feature=2020-resolver -e fedn/cli && \
    pip install --use-feature=2020-resolver -e fedn/sdk

FROM scaleout-fedn AS fed-bird-combiner

#COPY run.sh /app/
#COPY wsgi.py /app/
#COPY serve.py /app/
#COPY project.yaml /app/


