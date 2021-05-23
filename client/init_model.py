import sys
import os
sys.path.insert(0, '../../..')
from fedn.fedn.utils.kerashelper import KerasHelper
sys.path.insert(0, '../../../test/fedbird/client')

from kerasmodel import create_seed_model
from client_script import TrainingProcess, Model, TrainDataReader


if __name__ == '__main__':

    outfile_name = "../seed/birdweights.npz"
    helper = KerasHelper()

    start_process, _ = create_seed_model()

    helper.save_model(start_process.local_model.get_weights(), path=outfile_name)
    print("Seed model saved as: ", outfile_name)
