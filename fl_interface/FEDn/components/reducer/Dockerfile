#FROM debian:bullseye-slim AS platform-base
FROM nvcr.io/nvidia/tensorflow:20.08-tf2-py3 AS platform-base

#RUN apt-get update && \
#    apt-get install -y locales && \
#    localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8 

#ENV LANG en_US.utf8

#FROM platform-base AS system

#RUN apt update && \
#    apt install -y git adduser

FROM platform-base AS fedn-user

RUN adduser --disabled-password --gecos '' fednuser && \
    adduser fednuser sudo && \
    echo '%sudo ALL=(ALL:ALL) ALL' >> /etc/sudoers

#FROM fedn-user AS python-modules

#COPY requirements.txt requirements.txt
#RUN apt install -y python3 python3-pip
#	&& \
#    pip install -r requirements.txt

FROM fedn-user AS scaleout-fedn

WORKDIR /app/reducer

RUN git clone -b v0.1.0 https://github.com/scaleoutsystems/fedn.git && \
    pip install --use-feature=2020-resolver -e fedn/fedn

FROM scaleout-fedn AS fedn-bird-reducer

RUN mkdir certs && \
    chown -R fednuser /app && \
    rm -rf /var/lib/apt/lists/*

USER fednuser

