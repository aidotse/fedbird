import logging
import sys
import os
#sys.path.append('/media/sheetal/project_space/FL/code/fedbird')

if __name__ == '__main__':
    logger = logging.getLogger('__name__')
    logger.info("Calling the train function")
    from fedn.utils.kerasweights_modeltar import KerasWeightsHelper

    helper = KerasWeightsHelper()
    outer_model = helper.load_model(sys.argv[1])

    outer_model['model'].train('../data/Annotation/list1.txt')
    helper.save_model(outer_model, path=sys.argv[2])


