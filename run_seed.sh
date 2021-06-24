#  To run on client2 
# 1. Change the client1 to client2
# 2. Change farallon3 to roaster3

nvidia-docker run  -v  $PWD/client:/client -v $PWD/data:/data  -it  fedbird-farallon3-client python3 /client/init_model.py 

