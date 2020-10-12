import logging
import sys
#sys.path.append('/media/sheetal/project_space/FL/code/fedbird')

from client1_new import TrainingProcess, Model, TrainDataReader


if __name__ == '__main__':
    logger = logging.getLogger('__name__')
    logger.info("Calling the train function")
    model = Model()
    data = TrainDataReader()
    start_process = TrainingProcess(data,model,classes_path ='/app/client/model_data/seabird_classes.txt', anchors_path = '/app/client/model_data/tiny_yolo_anchors.txt', data_path ='/app/data/Annotation/list1.txt')
    #load global model weights here
    #global_model_weights = load_weights()
    local_model_weights = start_process.train(sys.argv[1])



