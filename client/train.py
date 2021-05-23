#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : train.py
# Author            : Sheetal Reddy <sheetal.reddy@ai.se>
# Date              : 08.03.2021
# Last Modified Date: 08.03.2021
# Last Modified By  : Sheetal Reddy <sheetal.reddy@ai.se>
import logging
import sys
from kerasmodel import create_seed_model
import os
#sys.path.append('/media/sheetal/project_space/FL/code/fedbird')

if __name__ == '__main__':
    logger = logging.getLogger('__name__')
    logger.info("Calling the train function")
    from fedn.utils.kerashelper import KerasHelper

    helper = KerasHelper()
    weights = helper.load_model(sys.argv[1])
    model = create_seed_model('.')
    model.local_model.set_weights(weights)
    model.train('/data/list1.txt','')
    helper.save_model(model.local_model.get_weights(), path=sys.argv[2])
