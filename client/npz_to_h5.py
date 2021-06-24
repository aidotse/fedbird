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
    weights = load_model(sys.argv[1])
    start_process = create_seed_model('/client/')
    start_process.set_weights(weights)
    start_process.save_weights(sys.argv[2])
