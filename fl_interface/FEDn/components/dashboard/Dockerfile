FROM python:3.8 AS platform-dashboard

WORKDIR /app

FROM platform-dashboard AS python-dashboard-modules

COPY requirements.txt requirements.txt
RUN apt-get update -y && \
    apt-get install -y python-pip python-dev && \
    pip install --use-feature=2020-resolver -r requirements.txt 

FROM python-dashboard-modules AS fedn-dashboard

