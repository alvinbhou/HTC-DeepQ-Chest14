from scipy.misc import imread, imsave, imresize
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from keras.utils import to_categorical
from keras.layers import Conv2D, BatchNormalization, Input, Activation
from keras.layers import LeakyReLU, Lambda, Reshape, Concatenate, Add, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler, Callback
from keras.optimizers import SGD, Adam
from keras import regularizers
import keras.backend as K
from keras.models import Model, load_model
import pickle
import numpy as np
from sklearn.externals import joblib
import cv2
import sys, random, time
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.resnet50 import ResNet50, preprocess_input

def test():
    with open('data/npy/X_valid.npy', 'rb') as f:
        X_valid = joblib.load(f)
    with open('data/npy/y_valid.npy', 'rb') as f:
        y_valid = joblib.load(f)
    model = load_model('model/new6_model.h5')
    scores = model.evaluate(X_valid, y_valid)
    print(scores)

test()