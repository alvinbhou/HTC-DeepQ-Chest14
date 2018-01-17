from scipy.misc import imread, imsave, imresize
import os,csv
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
from vis.utils import utils
from vis.visualization import visualize_saliency

data = {}
with open('data/Data_Entry_2017_v2.csv') as f:
    re = csv.reader(f)
    next(re)
    for r in re:
        id = r[0]
        dis = r[1].split("|")
        data[id] = dis

def test():
    with open('data/npy/X_valid.npy', 'rb') as f:
        X_valid = joblib.load(f)
    with open('data/npy/y_valid.npy', 'rb') as f:
        y_valid = joblib.load(f)
    print(X_valid.shape)


    model = load_model('models/noweight/resnet_no_weight_model.h5')
    layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == "dense_2"][0]

    b_model = load_model('models/bin1/model.h5')
    # model.summary()
    result = model.predict(X_valid)
    result_classes = result.argmax(axis=-1)
    # print(result)
    binary_result = []
    result2 = b_model.predict(X_valid)
    for x in result2:
        if  x[0] > 0.65:
            binary_result.append(1)
        else:
            binary_result.append(0)

    # binary_result is  1 = Infiltration, 0 = not Infiltration
    # need to binary_result with result_classes when output

    for idx, x in enumerate(X_valid):
        heatmap = visualize_saliency(model, layer_idx, np.expand_dims(result_classes[idx], axis=0), x)
        heatmap = None
        del heatmap
        
        # tracker.print_diff()
    

    
    
    # validset = {}
    # hasInfiltration = []
    # with open('data/pickles/labels_valid.pkl', 'rb') as f:
    #     validdata = pickle.load(f)
    #     for k, v in validdata.items(): 
    #         if(("Infiltration" in data[k])):
    #             hasInfiltration.append(1)
    #         else:
    #             hasInfiltration.append(0)
    
    # print(np.count_nonzero( np.array(result3) == np.array(hasInfiltration)))
   
        
    # print()
   

test()