import logging
import sys
import json
#sys.path.append('/media/sheetal/project_space/FL/code/fedbird')

if __name__ == '__main__':
    logger = logging.getLogger('__name__')
    logger.info("Calling the validate function")
    from fedn.utils.kerasweights_modeltar import KerasWeightsHelper

    helper = KerasWeightsHelper()
    outer_model = helper.load_model(sys.argv[1])

    results = outer_model['model'].validate('../data/Annotation/list1.txt')

    report = {
        #"classification_report": None,
        "training_loss": results[0],
        #"training_accuracy": 0.45,
        "test_loss": results[1],
        #"test_accuracy": 0.45,
    }

    with open(sys.argv[2], "w") as fh:
        fh.write(json.dumps(report))


