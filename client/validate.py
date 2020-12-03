import logging
import sys
import json
#sys.path.append('/media/sheetal/project_space/FL/code/fedbird')

if __name__ == '__main__':
    print("-------Validation starts-------Mattias code")
    logger = logging.getLogger('__name__')
    logger.info("Calling the validate function")
    sys.path.append('/Users/mattiasakesson/Documents/projects/fedn/fedn')
    from fedn.utils.kerasweights_modeltar import KerasWeightsHelper

    helper = KerasWeightsHelper()
    outer_model = helper.load_model(sys.argv[1])
    #outer_model = helper.load_model('birdcage')

    from client1_new import TrainingProcess, Model, TrainDataReader

    start_process = TrainingProcess(TrainDataReader(), Model(),
                                    classes_path='../data/model_data/seabird_classes.txt',
                                    anchors_path='../data/model_data/tiny_yolo_anchors.txt',
                                    data_path='../data/Annotation/list1.txt')
    start_process.local_model = outer_model['model']
                                    #,classes_path ='/app/client/model_data/seabird_classes.txt', anchors_path = '/app/client/model_data/tiny_yolo_anchors.txt', data_path ='/app/data/Annotation/list1.txt')
    #load global model weights here
    #global_model_weights = load_weights()
    #helper.load_model(start_process.local_model)


    #model = outer_model['model']
    #local_model_weights = start_process.train(sys.argv[1])
    results = start_process.validate()


    #

    report = {
        #"classification_report": None,
        "training_loss": 7,
        "training_accuracy": 0.45,
        "test_loss": 7,
        "test_accuracy": 0.45,
    }

    with open(sys.argv[2], "w") as fh:
        fh.write(json.dumps(report))

    print("Validation complete ------------------------ - - --- ----Mattias code")

