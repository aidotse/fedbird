import sys
import os

sys.path.append('/src/')

import keras
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
import tempfile

from src.client1_new import Model, TrainingProcess, TrainDataReader

if __name__ == '__main__':

	# Create a seed model and push to Minio
        m_instance = Model()
        data_instance = TrainDataReader()
        local_model = m_instance.build_model(data_instance.get_anchors(), len(data_instance.get_classes()))
        outfile_name = "test_model_fedbird"
        print('Seed model for Fedbird saved')
#       fod, outfile_name = tempfile.mkstemp(suffix='.h5') 

	#project = Project()
	#from scaleout.repository.helpers import get_repository

	#repo_config = {'storage_access_key': 'minio',
#				   'storage_secret_key': 'minio123',
#				   'storage_bucket': 'models',
#				   'storage_secure_mode': False,
#				   'storage_hostname': 'minio',
#				   'storage_port': 9000}
#	storage = get_repository(repo_config)

#	model_id = storage.set_model(outfile_name,is_file=True)
#	os.unlink(outfile_name)
#	print("Created seed model with id: {}".format(model_id))

	

