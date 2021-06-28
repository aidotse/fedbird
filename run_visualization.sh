#converts npz to h5 model
docker run  -v  $PWD/client:/client -v $PWD/data:/data  -it  fedbird-farallon3-client python3 /client/npz_to_h5.py /data/global_model /data/global_model.h5 

#uses the .h5 model and infers it on the video
nvidia-docker run  -v  $PWD/client:/client -v $PWD/data:/data  -it  fedbird-farallon3-client python3 /client/visualize.py --model /data/global_model.h5 --input /data/video.mp4 --output /data/output.mp4

