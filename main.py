from scipy.misc import imread, imsave, imresize
from keras.utils import to_categorical
from keras.layers import Conv2D, BatchNormalization, Input, Activation
from keras.layers import LeakyReLU, Lambda, Reshape, Concatenate, Add, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import keras.backend as K
from keras.models import Model
import pickle
import numpy as np
import os
import sys, random, time
from keras.applications.inception_resnet_v2 import InceptionResNetV2
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
CLASSES = 9
MODELDIR = 'models'
MODELFILE = 'model' + str(int(time.time())) +'.h5'
 
sv = ModelCheckpoint(os.path.join(MODELDIR, MODELFILE), save_best_only=True, save_weights_only=True)
es = EarlyStopping(patience=7)
 
def create_model_ResNetV2():
    INPUT_SIZE = 256
    xi = Input([INPUT_SIZE, INPUT_SIZE, 3 ])
    x = Reshape([INPUT_SIZE, INPUT_SIZE, 3])(xi)
 
    ir = InceptionResNetV2(
        include_top=False,
        weights= 'imagenet',
        input_tensor=x,
        input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
        pooling='avg')
   
    p = Dense(1000, activation='relu')(ir.output)    
    p = Dense(CLASSES, activation='sigmoid')(p)
 
    # ============ Model
    model = Model(ir.input, p)
    model.compile('adam', 'categorical_crossentropy', metrics= ['acc'])
    model.summary()
    return model
 
 
 
def x_gen(batch_size, valid = False):
    if(valid):
        with open('labels_valid.pkl', 'rb') as f:
            traindata = pickle.load(f)
    else:
        with open('labels_train.pkl', 'rb') as f:
            traindata = pickle.load(f)
    X, y = [], []
    while True:
        for k, v in traindata.items():
            if(v[0] == 0):
                continue
            img = imread('data/images/' + k, mode ='RGB')
            img = img / 255
            X.append(imresize(img ,size=(256,256)))
            y.append(v[0])
            if(len(X) == batch_size):
                y = to_categorical(y, num_classes=CLASSES)
                X = np.array(X)
                yield X,y
                X, y = [], []
 
def train():
    model = create_model_ResNetV2()
    model.fit_generator( x_gen(16, valid = False), validation_data = x_gen(16, valid = True),steps_per_epoch= 320, validation_steps = 14, epochs=50, callbacks=[es,sv])
   
   
train()