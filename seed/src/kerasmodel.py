import os

from client1_new import TrainingProcess, Model, TrainDataReader

def create_seed_model(path):
    model = Model()
    data = TrainDataReader()
    start_process = TrainingProcess(data, model, classes_path=os.path.join(path,'model_data/seabird_classes.txt'),
                 anchors_path=os.path.join(path,'model_data/tiny_yolo_anchors.txt'),
                 data_path=os.path.join(path,'data/Annotation/list1.txt'))

    return start_process.local_model