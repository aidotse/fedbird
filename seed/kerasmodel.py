
from client1_new import TrainingProcess, Model, TrainDataReader


def create_seed_model():
    model = Model()
    data = TrainDataReader()
    start_process = TrainingProcess(data, model)

    return start_process.local_model