#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from client1_new import TrainingProcess, Model, TrainDataReader

def create_seed_model(path, pretrained=False):
    model_instance = Model()
    data_reader  = TrainDataReader()
    start_process = TrainingProcess(data_reader, model_instance, load_pretrained_model=pretrained, pretrained_weights_path='/client/model_data/tiny.h5', classes_path=os.path.join(path,'model_data/seabird_classes.txt'),
                 anchors_path=os.path.join(path,'model_data/tiny_yolo_anchors.txt'))

    return start_process
