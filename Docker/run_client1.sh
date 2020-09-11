
ROOT_DIR=$PWD/..
DATA_DIR=$ROOT_DIR/data
CODE_DIR=$ROOT_DIR/src

nvidia-docker  run --rm  \
	-v $DATA_DIR:/data \
	-v $CODE_DIR:/src \
	-p 5656:5656 \
	--net host \
	-it fedml_bird \
	python3 client1.py
 




