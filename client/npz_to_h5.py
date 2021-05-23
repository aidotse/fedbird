import logging
import sys
import json
from kerasmodel import create_seed_model
from yolo import YOLO
from PIL import Image
import numpy as np

def load_model(path="weights.npz"):

        a = np.load(path)
        names = a.files
        weights = []
        for name in names:
            weights += [a[name]]

        return weights


if __name__ == '__main__':

    logger = logging.getLogger('__name__')
    logger.info("Calling the validate function")

    weights = load_model('model_data/global_model')
    model = create_seed_model('.')
    model.set_weights(weights)
    model.save_weights('model_data/global_model.h5')

