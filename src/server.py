__author__ = 'sheetal.reddy@ai.se'

"""Federated Learning Implementation v.0.1 - 2020-07-07 

This module provides classes and methods for FL server.
"""

import numpy as np
import os.path
import time
import pandas as pd
import shutil
import copy
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Lambda
from keras.optimizers import Adam
from keras import backend as K
from misc import get_classes, get_anchors
from yolo3.model import tiny_yolo_body, yolo_loss


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"]="1"
# A Simple multilayer perceptron neural network model (MLP)
class YOLOV3:
   @staticmethod
   def build(input_shape, anchors, classes, load_pretrained=True, freeze_body=2, weights_path='model_data/yolov3-tiny.h5'):
        '''create the training model, for Tiny YOLOv3'''
        K.clear_session() # get a new session
        image_input = Input(shape=(None, None, 3))
        h, w = input_shape
        num_anchors = len(anchors)

        y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
            num_anchors//2, num_classes+5)) for l in range(2)]

        model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
        print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

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
        model = Model([model_body.input, *y_true], model_loss)

        return model
    
def avr_weights(weight_list):
    """Get the average weight accross all client gradients
    
    Parameters    ----------
    weight_list : list of all the clients weights
    
    Returns
    -------
    avg_grad : average weight accross all client
    
    """
    w_avg = np.array(copy.deepcopy(weight_list[0]))
    print(np.array(weight_list[0][1]).shape)
    #for grad_list_tuple in zip(*weight_list):
    #    layer_mean = tf.math.reduce_mean(grad_list_tuple, axis=0)
    #    avg_grad.append(layer_mean)
    for j in range(len(weight_list[0])): # no of layers
        for i in range(1, len(weight_list)): # no of clients
            w_avg[j] +=np.array(weight_list[i][j])
        w_avg[j] = w_avg[j] / len(weight_list)    
    return w_avg

def clear_dir(List_dir):
    """Clear folders of server and clients models
    
    Parameters
    ----------
    List_dir : specific directory that is defined to save global model or local models for each client
    
    """
    
    for i in List_dir:
        for root, dirs, files in os.walk(i):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))


def main(global_model, iteration_num, Server_dir, Clients_dir, epoch):
    """Server main function to read the local model files, aggregate the weights, and update the global model
    
    Parameters
    ----------
    global_model : NN model with class SimpleMLP
    iteration_num : number of iteration 
    Server_dir : specific directory that is defined to only save global model
    Clients_dir : specific directory that is defined to only save local models for each client
    
    """
    
    print('>>>>>>>>>>     FL for %d iteration with %d clients     <<<<<<<<<<' %(iteration_num, len(Clients_dir)))
    
    for itr in range(0,iteration_num, epoch):
        print('Iteration number:', itr)

        #initial list to collect local model weights 
        local_weight_list = list()


        # for each client if the folder is not empty and if epoch number matches the common_round
        # add the local weight to the local_weight_list
        for cl in range(len(Clients_dir)):
            while True: 
                data_file = "".join([Clients_dir[cl],'client_data.csv'])
                while not os.path.exists(data_file):
                    time.sleep(2)
                local_data_file = pd.read_csv(data_file)
                if (int(local_data_file['epoch'])) == itr+epoch:
                    print('local data received from:', local_data_file['name'][0])
                    break;
            weights_file = "".join([Clients_dir[cl],'local_model.h5'])
            #global_model = keras.models.load_model(model_file)
            global_model.load_weights(weights_file)
            local_weight_list.append(global_model.get_weights())
        #to get the average over all the local model, we simply take the sum of the scaled weights
        average_weights = avr_weights(local_weight_list)

        # set the gobal_model weights with the average weight from each clients
        global_model.set_weights(average_weights)

        # save the server_data and global model
        global_model.save_weights("".join([Server_dir,'global_model.h5']), overwrite=True)
        server_data = pd.DataFrame({'name': ['Server'], 'epoch': [itr+epoch], 'iteration_num': [iteration_num]})
        server_data.to_csv("".join([Server_dir,'server_data.csv']), index=False)
        #tf.keras.models.save_model(global_model, "".join([Server_dir,'global_model']), overwrite=True)
        #tf.saved_model.save(global_model, "".join([Server_dir,'global_model']))


if __name__ == "__main__":
    # clear folders of server and clients models    
    Clients_dir = ['/data/CL1/','/data/CL2/']
    for folder in Clients_dir:
        if os.path.isdir(folder):
            clear_dir(Clients_dir)
        else:
            os.mkdir(folder)
    
    Server_dir = ['/data/CLS/']
    for folder in Server_dir :
        if os.path.isdir(folder):
            clear_dir(Server_dir)
        else:
            os.mkdir(folder)
     
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
           for gpu in gpus:
               tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    print('Server and Client folders are clered ......')
    
    annotation_path = 'train.txt'
    classes_path = 'model_data/seabird_classes.txt'
    anchors_path = 'model_data/tiny_yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (416,416) # multiple of 32, hw

    # initialization of golab model
    smlp_global = YOLOV3()
    global_model = smlp_global.build(input_shape,anchors,num_classes)
    #model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
    #        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
    #        [*model_body.output, *y_true])

    #global_model.add_loss(model_loss) 
    lr = 0.001 
    iteration_num = 30
    epoch = 1
    #loss={'yolo_loss': lambda y_true, y_pred: y_pred}
    #optimizer = Adam(lr=lr, decay=lr / iteration_num)
    #global_model.compile(loss=loss, optimizer=optimizer)
    #global_model.build(input_shape)
    #tf.saved_model.save(global_model, "/data/CLS/global_model")
    global_model.save_weights( "/data/CLS/global_model.h5")
    

    server_data = pd.DataFrame({'name': ['Server'], 'epoch': [0], 'iteration_num': [iteration_num]})
    server_data.to_csv('/data/CLS/server_data.csv', index=False)

    main(global_model, iteration_num, Server_dir[0], Clients_dir, epoch)

