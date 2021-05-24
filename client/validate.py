#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import logging
import sys
import json
from kerasmodel import create_seed_model


if __name__ == '__main__':

    logger = logging.getLogger('__name__')
    logger.info("Calling the validate function")
    from fedn.utils.kerashelper import KerasHelper

    helper = KerasHelper()
    weights = helper.load_model(sys.argv[1])
    model = create_seed_model('.')
    model.set_weights(weights)

    results = model.validate('/data/val.txt','')
    report = {
        "training_loss": results[0],
        "test_loss": results[1],
        'mAP'      : results[2]
    }

    with open(sys.argv[2], "w") as fh:
        fh.write(json.dumps(report))



