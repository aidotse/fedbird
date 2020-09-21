FROM python:3.8 as platform-discovery

WORKDIR /app

FROM platform-discovery AS python-modules

COPY requirements.txt requirements.txt
RUN pip install --use-feature=2020-resolver -r requirements.txt 

FROM python-modules AS fedn-discovery


