FROM python:3.8 AS monitor-base

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --use-feature=2020-resolver -r requirements.txt 


FROM monitor-base AS scaleout-fedn

RUN git clone https://github.com/scaleoutsystems/fedn.git && \
    pip install --use-feature=2020-resolver -e fedn/cli && \
    pip install --use-feature=2020-resolver -e fedn/sdk

FROM scaleout-fedn AS fedn-bird-monitor
