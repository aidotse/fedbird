import logging
import sys
#sys.path.append('/media/sheetal/project_space/FL/code/fedbird')

if __name__ == '__main__':
    logger = logging.getLogger('__name__')
    logger.info("Calling the train function")
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
                                   
    local_model = start_process.train()
    outer_model['model'] = start_process.local_model
    helper.save_model(outer_model, path=sys.argv[2])


