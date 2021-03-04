import logging
import sys
import json
from kerasmodel import create_seed_model

#sys.path.append('/media/sheetal/project_space/FL/code/fedbird')

if __name__ == '__main__':

    logger = logging.getLogger('__name__')
    logger.info("Calling the validate function")
    from fedn.utils.kerashelper import KerasHelper

    helper = KerasHelper()
    weights = helper.load_model(sys.argv[1])
    model = create_seed_model('.')
    model.set_weights(weights)
    #model.save_weights('model_data/global_model.h5', overwrite=True)

    results = model.validate('/data/list1_fedn.txt','')
    report = {
        "training_loss": results[0],
        "test_loss": results[1],
        # "training_AUC": results[2],
        # "test_AUC": results[3],
        # "training_Precision": results[4],
        # "test_Precision": results[5],
        # "training_Recall": results[6],
        # "test_Recall": results[7],
        # "train_iningVOC_PASCAL_mAP": results[2],
        # "test_VOC_PASCAL_mAP": results[3],
        # "training_COCO_mAP": results[4],
        # "test_COCO_mAP": results[5],
    }

    with open(sys.argv[2], "w") as fh:
        fh.write(json.dumps(report))



