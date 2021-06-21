#  To run on client2 
# 1. Change the client1 to client2
# 2. Change farallon3 to roaster3


nvidia-docker run --gpus all --add-host=combiner:172.25.16.6  -v  $PWD/data/client1:/data fedbird-farallon3-client /bin/bash -c "fedn run client -in fedn-network-xavier.yaml"

