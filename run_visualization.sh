
#converts npz to h5 model
#1. place your mpz model in model/ directory and change the path /model/global_model to /model/<model name>
docker run  -v  $PWD/client:/client -v $PWD/model:/model  -it  fedbird-farallon3-client python3 /client/npz_to_h5.py /model/global_model /model/global_model.h5 



#uses the .h5 model and infers it on the video
#1. Change the model path, input video, output video when ever needed.
nvidia-docker run  -v  $PWD/client:/client -v $PWD/video:/video  -v $PWD/model:/model -it  fedbird-farallon3-client python3 /client/visualize.py --model /model/global_model.h5 --input /video/video.mp4 --output /video/output.mp4

