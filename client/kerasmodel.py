#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : kerasmodel.py
# Author            : Sheetal Reddy <sheetal.reddy@ai.se>
# Date              : 10.03.2021
# Last Modified Date: 10.03.2021
# Last Modified By  : Sheetal Reddy <sheetal.reddy@ai.se>
import os
from client1_new import TrainingProcess, Model, TrainDataReader

def create_seed_model(path):
    model = Model()
    data = TrainDataReader()
    start_process = TrainingProcess(data, model, classes_path=os.path.join(path,'model_data/seabird_classes.txt'),
                 anchors_path=os.path.join(path,'model_data/tiny_yolo_anchors.txt'),
                 data_path='/data/list1.txt',
                 data_root_path='')

    return start_process
