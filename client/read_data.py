import numpy
import pandas as pd
from sklearn.model_selection import train_test_split

import keras
import numpy

def read_data(filename, nr_examples=1000,bias=0.7):
    """ Helper function to read and preprocess data for training with Keras. """

    data = numpy.array(pd.read_csv(filename))
    X = data[:, 1::]
    y = data[:, 0]

    sample_fraction = float(nr_examples)/len(y)  

    # The entire dataset is 60k images, we can subsample here for quicker testing. 
    if sample_fraction < 1.0:
        _, X, _, y = train_test_split(X, y, test_size=sample_fraction)
    classes = range(10)

    # Input image dimensions
    img_rows, img_cols = 28, 28

    # The data, split between train and test sets
    X = X.reshape(X.shape[0], img_rows, img_cols, 1)
    X = X.astype('float32')
    X /= 255
    y = keras.utils.to_categorical(y, len(classes))
    return  (X, y, classes)


