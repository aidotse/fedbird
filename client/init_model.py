#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : init_model.py
# Author            : Sheetal Reddy <sheetal.reddy@ai.se>
# Date              : 08.03.2021
# Last Modified Date: 08.03.2021
# Last Modified By  : Sheetal Reddy <sheetal.reddy@ai.se>

import sys
import os
from client1_new import TrainingProcess, Model, TrainDataReader
from fedn.utils.kerashelper import KerasHelper

def create_seed_model():
    model = Model()
    data = TrainDataReader()
    start_process = TrainingProcess(data, model,
                                    load_pretrained_model = True,
                                    pretrained_weights_path ='../client/model_data/tiny.h5',
                                    classes_path='../client/model_data/seabird_classes.txt',
                                    anchors_path='../client/model_data/tiny_yolo_anchors.txt')
                                    #data_path='../data/Annotation/list1.txt')

    return start_process#.local_model



if __name__ == '__main__':

    outfile_name = "/data/birdweights.npz"

    helper = KerasHelper()
    model = create_seed_model()
    helper.save_model(model.local_model.get_weights(), path=outfile_name)
    print("seed model saved as: ", outfile_name)
