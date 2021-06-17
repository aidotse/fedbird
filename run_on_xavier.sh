nvidia-docker run --gpus all --add-host=combiner:172.25.16.6  -v  $PWD/data/client1:/data fedbird-farallon3-client /bin/bash -c "fedn run client -in fedn-network-xavier.yaml"

