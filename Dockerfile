#To run on cpu use below
FROM python:3.6.8

RUN pip install --upgrade pip 

RUN pip install -e git://github.com/scaleoutsystems/fedn.git@v0.2.3#egg=fedn\&subdirectory=fedn

# Get latest version of mean_average_precision library for object detection
RUN pip3 install --upgrade git+https://github.com/bes-dev/mean_average_precision.git
RUN pip3 install mean_average_precision
COPY fedn-network.yaml /app/ 
COPY requirements.txt /app/
WORKDIR /app
RUN pip install -r requirements.txt
