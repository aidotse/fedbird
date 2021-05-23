
"""Federated Learning Implementation v.0.1 - 2020-10-06 

This module provides classes and methods for FL clients.
"""

import numpy as np
import random
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
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from mean_average_precision import MetricBuilder
from keras.layers import Input, Lambda
from keras.optimizers import Adam
from keras import backend as K
from yolo3.model import tiny_yolo_body, yolo_loss, preprocess_true_boxes, yolo_eval
from yolo3.utils import get_random_data, get_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


class Model:

    def __init__(self, input_shape=(416, 416), model_path=None):

        self.input_shape = input_shape
        self.current_model = None
        self.model_body = None 
        self.logger = logging.getLogger('__name__')

    def build_model(self, anchors, num_classes, load_pretrained=False, freeze_body=2,
                    weights_path='./model_data/tiny.h5'):
        '''create the training model, for Tiny YOLOv3'''
        K.clear_session()  # get a new session
        image_input = Input(shape=(None, None, 3))
        h, w = self.input_shape
        num_anchors = len(anchors)

        y_true = [Input(shape=(h // {0: 32, 1: 16}[l], w // {0: 32, 1: 16}[l], \
                               num_anchors // 2, num_classes + 5)) for l in range(2)]

        self.model_body = tiny_yolo_body(image_input, num_anchors // 2, num_classes)
        self.logger.info('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

        if load_pretrained:
            self.model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
            print('Load weights {}.'.format(weights_path))
            if freeze_body in [1, 2]:
                # Freeze the darknet body or freeze all but 2 output layers.
                num = (20, len(self.model_body.layers) - 2)[freeze_body - 1]
                for i in range(num): self.model_body.layers[i].trainable = False
                print('Freeze the first {} layers of total {} layers.'.format(num, len(self.model_body.layers)))

        model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                            arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
            [*self.model_body.output, *y_true])
        self.current_model = keras.models.Model([self.model_body.input, *y_true], model_loss)
        return self.current_model

    def get_model_body(self):
       return self.model_body



class TrainDataReader:

    # parameters initializiation
    def __init__(self):
        self.batch_size = 32
        self.logger = logging.getLogger('__name__')

        # get the data from files

    def read_training_data(self, data_root_path="..", annotation_path=None , val_split=0.1):
        with open(annotation_path) as f:
            lines = f.readlines()
        lines = [data_root_path + line for line in lines]
        np.random.seed(10101)
        np.random.shuffle(lines)
        np.random.seed(None)
        num_val = int(len(lines) * val_split)
        num_train = len(lines) - num_val
        self.logger.debug('num_train and num_val values' + str(num_train) + str(num_val))
        # print('num_train and num_val values'+ str(num_train),str(num_val))
        return lines[:num_train], lines[num_train:]

    # def get_all_data(self,annotation_lines, input_shape, anchors, num_classes):
    def get_gt(self, annotation_lines, input_shape, num_classes):
        '''geenrates gt boxes for map calculation'''
        n = len(annotation_lines)
        gts = []
        image_data = []
        for i in range(0, n):
            image, box = get_data(annotation_lines[i], input_shape, random=False)
            image_data.append(image)
            gts.append(box)
        image_data = np.array(image_data)

        return image_data,gts

    def data_generator(self, annotation_lines, input_shape, anchors, num_classes):
        '''data generator for fit_generator'''
        n = len(annotation_lines)
        i = 0
        while True:
            image_data = []
            box_data = []
            for b in range(self.batch_size):
                if i == 0:
                    np.random.shuffle(annotation_lines)

                #annot_path = os.path.join('/app', annotation_lines[i][1:])
                # annot_path = os.path.join('../..', annotation_lines[i][1:])
                #print("annotation_lines: ", annotation_lines)
                image, box = get_random_data(annotation_lines[i], input_shape, random=True)
                image_data.append(image)
                box_data.append(box)
                i = (i + 1) % n
            image_data = np.array(image_data)
            box_data = np.array(box_data)
            y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
            yield [image_data, *y_true], np.zeros(self.batch_size)

    def data_generator_wrapper(self, annotation_lines, input_shape, anchors, num_classes):
        n = len(annotation_lines)
        if n == 0 or self.batch_size <= 0: return None
        return self.data_generator(annotation_lines, input_shape, anchors, num_classes)

    def get_classes(self, classes_path='model_data/seabird_classes.txt'):
        '''loads the classes'''
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def get_anchors(self, anchors_path='model_data/tiny_yolo_anchors.txt'):
        '''loads the anchors from a file'''
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)


class TrainingProcess:

    def __init__(self, data, model, input_shape=(416, 416), learningrate=1e-3, epoch=1,
                 classes_path='model_data/seabird_classes.txt',
                 anchors_path='model_data/tiny_yolo_anchors.txt',
                 load_pretrained_model=False,
                 pretrained_weights_path =None, 
                 data_path=None,
                 data_root_path=None):

        self.init_epoch = 0
        self.epoch = epoch
        self._data = data
        self._model = model
        self.anchors_path = anchors_path
        self.classes_path = classes_path
        self.class_names = self._data.get_classes(self.classes_path)
        self.num_classes = len(self.class_names)
        self.anchors = self._data.get_anchors(self.anchors_path)
        self.input_shape = input_shape
        self.load_pretrained = load_pretrained_model
        self.lr = learningrate
        if data_path:
            self.lines_train, self.lines_val = self._data.read_training_data(data_root_path,data_path)
        else:
            self.lines_train, self.lines_val = None, None
        self.log_dir = 'logs/000/'
        self.logging = TensorBoard(log_dir=self.log_dir)
        self.checkpoint = ModelCheckpoint(self.log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                          monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
        self.early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
        self.local_model = self._model.build_model(self.anchors,
                                                   self.num_classes, load_pretrained=self.load_pretrained, weights_path =pretrained_weights_path)  # model created, self._model.current_model
        self.metric_fn = MetricBuilder.build_evaluation_metric( "map_2d", async_mode=True ,num_classes=self.num_classes)
        
        self.model_body = self._model.get_model_body()
        self.input_image_shape = K.placeholder(shape=(2, ))
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = yolo_eval(self.model_body.output, self.anchors,
                             self.num_classes,self.input_image_shape, score_threshold=0.3, iou_threshold=0.5)

        
        self.logger = logging.getLogger('FedBird')
        # self.local_model = None

    def save_weights(self,path):
        self.local_model.save_weights(path)

    def load_weights(self,path):
        self.local_model.load_weights(path)


    def set_weights(self, weights):
        self.local_model.set_weights(weights)


    def get_weights(self):
        return self.local_model.get_weights()

    def train(self, data_path=None, data_root_path=".."):
        """Client main function to read the global model files, update the weights, and save the local model

        Parameters
        ----------

        """
        if data_path:
            self.lines_train, self.lines_val = self._data.read_training_data(data_root_path,data_path)
        end_epoch = self.init_epoch + self.epoch

        while True:
            # fit local model with client's data

            if self.init_epoch < 7:
                self.logger.info('Freezing the initial layers')
                self.local_model.compile(optimizer=Adam(lr=self.lr), loss={  # use custom yolo_loss Lambda layer.
                    'yolo_loss': lambda y_true, y_pred: y_pred})
            else:
                self.logger.info('Unfreezing all the layers')
                for i in range(len(self.local_model.layers)):
                    self.local_model.layers[i].trainable = True
                self.local_model.compile(optimizer=Adam(lr=self.lr),
                                         loss={'yolo_loss': lambda y_true,
                                                                   y_pred: y_pred})

            # batch_size = 32 # note that more GPU memory is required after unfreezing the body
            self.logger.info('Train on {} samples, val on {} samples, with batch size {}.'
                             .format(len(self.lines_train), len(self.lines_val), self._data.batch_size))
            self.local_model.fit_generator(self._data.data_generator_wrapper(self.lines_train,
                                                                             self.input_shape, self.anchors,
                                                                             self.num_classes),
                                           steps_per_epoch=max(1, len(self.lines_train) // self._data.batch_size),
                                           validation_data=self._data.data_generator_wrapper(self.lines_val,
                                                                                             self.input_shape,
                                                                                             self.anchors,
                                                                                             self.num_classes),
                                           validation_steps=max(1, len(self.lines_val) // self._data.batch_size),
                                           epochs=end_epoch,
                                           initial_epoch=self.init_epoch,
                                           callbacks=[self.logging, self.checkpoint])
            # clear session to free memory after each communication round
            self.init_epoch += self.epoch
            #self.save_weights('./model_data/local_model.h5', overwrite=True)
            self.local_model.save_weights('./model_data/llllocal_model.h5', overwrite=True)
            if self.init_epoch == end_epoch:
                self.logger.info('Local Training Completed')
                return self.local_model

    def validate(self, data_path=None, data_root_path=".."):
        """Client main function to read the global model files, update the weights, and save the local model

        Parameters
        ----------

        """
        if data_path:
            _, self.lines_val = self._data.read_training_data(data_root_path, data_path, val_split=1)

        self.local_model.compile(optimizer=Adam(lr=self.lr),
                                 loss={'yolo_loss': lambda y_true, y_pred: y_pred})
                                 #metrics=['AUC', 'Precision', 'Recall'])  # recompile to apply the change
        # batch_size = 32 # note that more GPU memory is required after unfreezing the body
        self.logger.info('Train on {} samples, val on {} samples, with batch size {}.'
                         .format(len(self.lines_train), len(self.lines_val), self._data.batch_size))
        train_results = self.local_model.evaluate_generator(
            self._data.data_generator_wrapper(self.lines_train, self.input_shape, self.anchors, self.num_classes),
            steps=max(1, len(self.lines_train) // self._data.batch_size))
        val_results = self.local_model.evaluate_generator(
            self._data.data_generator_wrapper(self.lines_val, self.input_shape, self.anchors, self.num_classes),
            steps=max(1, len(self.lines_val) // self._data.batch_size))

        # calculate map here
        images_data, gt_boxes = self._data.get_gt(self.lines_val, self.input_shape, self.num_classes)
        # Generate output tensor targets for filtered bounding boxes.
        for j in range(0,images_data.shape[0]):
            preds = []
            image_data = images_data[j,:]
            gt = gt_boxes[j]
            image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
            out_boxes, out_scores, out_classes = self.sess.run(
                                [self.boxes, self.scores, self.classes],
            feed_dict={
                self.model_body.input: image_data,
                self.input_image_shape: [416, 416],
                K.learning_phase(): 0
            })

            #print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

            # change the format of the output
            for i, c in reversed(list(enumerate(out_classes))):
                predicted_class = self.class_names[c]
                box = out_boxes[i]
                score = out_scores[i]

                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(416, np.floor(bottom + 0.5).astype('int32'))
                right = min(416, np.floor(right + 0.5).astype('int32'))
                preds.append([left, top, right, bottom, int(c), float(score)])
            preds=np.array(preds)
            self.metric_fn.add(preds,gt)
        mean_ap = self.metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.1, 0.1))['mAP']
        results = [train_results, val_results, str(mean_ap) ]


        return results


if __name__ == "__main__":
    m_instance = Model()
    start_process = TrainingProcess(TrainDataReader(), m_instance, epoch=1)
    final_model = start_process.train('../data/client2/list.txt')
    results = start_process.validate('../data/client2/list.txt')

