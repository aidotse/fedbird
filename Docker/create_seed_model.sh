
ROOT_DIR=$PWD/..
DATA_DIR=$ROOT_DIR/data
CODE_DIR=$ROOT_DIR

nvidia-docker  run   \
	-v $DATA_DIR:/data \
	-v $CODE_DIR:/src \
	-p 5656:5656 \
	--net host \
	-it fedml_bird \
	python3 fl_interface/FEDn/components/seed/init_model_fedbird.py



