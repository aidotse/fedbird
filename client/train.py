#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import sys
from kerasmodel import create_seed_model
import os

if __name__ == '__main__':
    logger = logging.getLogger('__name__')
    logger.info("Calling the train function")
    from fedn.utils.kerashelper import KerasHelper

    helper = KerasHelper()
    weights = helper.load_model(sys.argv[1])
    model = create_seed_model('.')
    model.local_model.set_weights(weights)
    model.train('/data/train.txt','')
    helper.save_model(model.local_model.get_weights(), path=sys.argv[2])
