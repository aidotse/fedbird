import logging
import sys
sys.path.append('/media/sheetal/project_space/FL/code/fedbird')


from src.client1_new import TrainingProcess, Model, TrainDataReader


if __name__ == '__main__':
    logger = logging.getLogger('__name__')
    logger.info("Calling the train function")
    model = Model()
    data = TrainDataReader()
    start_process = TrainingProcess(data,model)
    #load global model weights here
    #global_model_weights = load_weights()
    local_model_weights = start_process.train(global_model_weights)



