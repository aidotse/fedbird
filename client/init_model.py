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
from kerasmodel import create_seed_model


if __name__ == '__main__':

    outfile_name = '/seed/'+ sys.argv[1]
    helper = KerasHelper()
    start_process = create_seed_model('/client/')
    helper.save_model(start_process.local_model.get_weights(), path=outfile_name)
    print("seed model saved as: ", outfile_name)
