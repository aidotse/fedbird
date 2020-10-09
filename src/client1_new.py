__author__ = 'sheetal.reddy@ai.se'

"""Federated Learning Implementation v.0.1 - 2020-10-06 

This module provides classes and methods for FL clients.
"""

import numpy as np
import random
import cv2
import os
import pandas as pd
import time
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

import tensorflow as tf
import keras
#from keras.models import Model 
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from keras.layers import Input, Lambda
from keras.optimizers import Adam
from keras import backend as K
from src.yolo3.model import tiny_yolo_body, yolo_loss, preprocess_true_boxes
from src.yolo3.utils import get_random_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

class Model:

   def __init__(self,input_shape=(416,416)):

      self.input_shape = input_shape
      self.current_model = None
      self.logger = logging.getLogger('__name__')

   def build_model(self, anchors, num_classes, load_pretrained=False, freeze_body=2, weights_path='src/model_data/yolov3-tiny.h5'):
        '''create the training model, for Tiny YOLOv3'''
        K.clear_session() # get a new session
        image_input = Input(shape=(None, None, 3))
        h, w = self.input_shape
        num_anchors = len(anchors)

        y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
            num_anchors//2, num_classes+5)) for l in range(2)]

        model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
        self.logger.info('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

        if load_pretrained:
            model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
            print('Load weights {}.'.format(weights_path))
            if freeze_body in [1, 2]:
                # Freeze the darknet body or freeze all but 2 output layers.
                num = (20, len(model_body.layers)-2)[freeze_body-1]
                for i in range(num): model_body.layers[i].trainable = False
                print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

        model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
            arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
            [*model_body.output, *y_true])
        self.current_model = keras.models.Model([model_body.input, *y_true], model_loss)
        #self.current_model.save('test_model')
        return self.current_model


class TrainDataReader:
    
    # parameters initializiation
    def __init__(self):
      #self.annotation_path = annotation_path
      self.batch_size = 32
      self.logger = logging.getLogger('__name__') 
    
    # get the data from files
    def read_training_data(self, annotation_path= '/data/Annotation/list1.txt',val_split = 0.1):
        with open(annotation_path) as f:
            lines = f.readlines()
        np.random.seed(10101)
        np.random.shuffle(lines)
        np.random.seed(None)
        num_val = int(len(lines)*val_split)
        num_train = len(lines) - num_val
        self.logger.debug('num_train and num_val values'+ str(num_train)+str(num_val))
        print('num_train and num_val values'+ str(num_train),str(num_val))
        return lines[:num_train], lines[num_train:]

    def data_generator(self,annotation_lines, input_shape, anchors, num_classes):
        '''data generator for fit_generator'''
        n = len(annotation_lines)
        i = 0
        while True:
            image_data = []
            box_data = []
            for b in range(self.batch_size):
                if i==0:
                    np.random.shuffle(annotation_lines)
                image, box = get_random_data(annotation_lines[i], input_shape, random=True)
                image_data.append(image)
                box_data.append(box)
                i = (i+1) % n
            image_data = np.array(image_data)
            box_data = np.array(box_data)
            y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
            yield [image_data, *y_true], np.zeros(self.batch_size)

    def data_generator_wrapper(self, annotation_lines, input_shape, anchors, num_classes):
        n = len(annotation_lines)
        if n==0 or self.batch_size<=0: return None
        return self.data_generator(annotation_lines,input_shape, anchors, num_classes)
   
    def get_classes(self, classes_path ='src/model_data/seabird_classes.txt'):
        '''loads the classes'''
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def get_anchors(self, anchors_path='src/model_data/tiny_yolo_anchors.txt'):
        '''loads the anchors from a file'''
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)



class TrainingProcess:

    def __init__(self, data, model, input_shape = (416,416), learningrate=1e-3, epoch=1, classes_path='src/model_data/seabird_classes.txt', anchors_path = 'src/model_data/tiny_yolo_anchors.txt'):

        self.init_epoch = 0
        self.epoch = epoch
        self._data = data
        self._model = model
        self.anchors_path = anchors_path
        self.classes_path = classes_path
        self.class_names = _data.get_classes(self.classes_path)
        self.num_classes = len(self.class_names)
        self.anchors = _data.get_anchors(self.anchors_path)
        self.input_shape = input_shape
        self.lr = learningrate
        self.lines_train, self.lines_val = self._data.read_training_data()
        self.log_dir = 'logs/000/'
        self.logging = TensorBoard(log_dir=self.log_dir)
        self.checkpoint = ModelCheckpoint(self.log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                    monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
        self.early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
        self.local_model = self._model.build_model(self.anchors, self.num_classes) # model created, self._model.current_model
        self.logger = logging.getLogger('FedBird')    
      
    def train(self, global_model):
        """Client main function to read the global model files, update the weights, and save the local model

        Parameters
        ----------

        """
        #update it with the global model?
        end_epoch = self.init_epoch + self.epoch
        #local_model = keras.models.clone_model(global_model)
        self.local_model.load_weights(global_model) # have to fix what is this global_model
        while True:
            # fit local model with client's data
             #local_model.fit(clients_batched, epochs=epoch, verbose=1)
            if (self.init_epoch<50):
                 self.logger.info('Freezing the initial layers')
                 self.local_model.compile(optimizer=Adam(lr=self.lr), loss={ # use custom yolo_loss Lambda layer.
                                     'yolo_loss': lambda y_true, y_pred: y_pred})
            else:
                 self.logger.info('Unfreezing all the layers')
                 for i in range(len(local_model.layers)):
                     self.local_model.layers[i].trainable = True
                 self.local_model.compile(optimizer=Adam(lr=self.lr), 
                         loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change

            #batch_size = 32 # note that more GPU memory is required after unfreezing the body
            self.logger.info('Train on {} samples, val on {} samples, with batch size {}.'
                    .format(len(lines_train), len(lines_val), self._data.batch_size))
            self.local_model.fit_generator(self._data.data_generator_wrapper(lines_train,
                         self.input_shape, self.anchors, self.num_classes),
                     steps_per_epoch=max(1, len(lines_train)//self._data.batch_size),
                     validation_data=self._data.data_generator_wrapper(lines_val, 
                         self.input_shape, self.anchors, self.num_classes),
                     validation_steps=max(1, len(lines_val)//self._data.batch_size),
                     epochs= end_epoch,
                     initial_epoch= self.init_epoch,
                     callbacks=[self.logging,self.checkpoint])
                 # clear session to free memory after each communication round
            self.init_epoch += self.epoch
            self.local_model.save_weights('local_model.h5', overwrite=True)
            if self.init_epoch == end_epoch:
                self.logger.info('Local Training Completed')
                return self.local_model

      
if __name__ == "__main__":
    
    m_instance=Model()
    start_process = TrainingProcess(TrainDataReader(),m_instance)
    model = keras.models.load_model("test_model")
    final_model = start_process.train(model)
    
   
