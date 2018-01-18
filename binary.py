from scipy.misc import imread, imsave, imresize
import os
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
LABELS = {
    'No Finding': -1,
    'Atelectasis': 1,
    'Cardiomegaly': 2,
    'Consolidation': -1,
    'Edema': -1,
    'Effusion': 3,
    'Emphysema': -1,
    'Fibrosis': -1,
    'Hernia': -1,
    'Infiltration': 4,
    'Infiltrate': -1,
    'Mass': 5,
    'Nodule': 6,
    'Pleural_Thickening': -1,
    'Pneumonia': 7,
    'Pneumothorax': 8
}
trainset = set()

class LogCallback(Callback):
    def __init__(self):
        super(Callback, self).__init__()
        

    def on_epoch_end(self, epoch, logs={}):
        with open(os.path.join(MODELDIR, 'log.txt'), 'a') as file:
            file.write(str(epoch) + ',' + str(logs['loss']) +',' + str(logs['acc']) + ',')
            file.write(str(logs['val_loss']) + ',' + str(logs['val_acc']) + '\n')

model_id = sys.argv[1]
if not (os.path.exists("models/%s/" % model_id)):
    os.makedirs("models/%s/" % model_id)
MODELDIR = os.path.join("models", model_id)
MODELFILE = 'model.h5'
CLASSES =  1

sv = ModelCheckpoint(os.path.join(MODELDIR, MODELFILE),
                            monitor='val_acc',
                            verbose=1,
                            save_best_only=True,
                            save_weights_only=False,
                            mode='auto',
                            period=1)



def create_base_model( w = 'imagenet', trainable = False):
    model = ResNet50(weights=w, include_top=False, input_shape=(224, 224, 3))
    if(not trainable):
        for layer in model.layers:
            layer.trainable = False
    return model

def rebase_base_model(model):
    # for layer in model.layers[:self.num_fixed_layers]:
    #     layer.trainable = False
    for layer in model.layers:
        layer.trainable = True
    return model

def add_custom_layers(base_model):
        x = base_model.output
        x = Flatten()(x)
        # x = Dropout(0.2)(x)
        # x = Dense(1024, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.005))(x)
        x = Dense(2048, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.005))(x)
        y = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=base_model.input, outputs=y)
        return model


def generator(batch_size, valid = False):
    DATA_ROOT_PATH = 'data/bin'
    while True:
        if(valid):
            with open('data/bin/X_5.npy', 'rb') as f:
                X = joblib.load(f)
            with open('data/bin/y_5.npy', 'rb') as f:
                y = joblib.load(f)


            # with open(os.path.join(DATA_ROOT_PATH, 'X_11' + '.npy'), 'rb') as file:
            #     X = joblib.load(file)
            # with open(os.path.join(DATA_ROOT_PATH, 'y_11' + '.npy'), 'rb') as file:
            #     y = joblib.load(file) 
            # X = np.array(X)
            # y = to_categorical(y, num_classes=CLASSES)
            for j in range((len(X) // batch_size) - 1):
                    yield X[j * batch_size: (j+1) * batch_size], y[ j * batch_size: (j+1) * batch_size]

        else:
            for i in range(1,5):
                with open(os.path.join(DATA_ROOT_PATH, 'X_' + str(i) + '.npy'), 'rb') as file:
                    X = joblib.load(file)
                with open(os.path.join(DATA_ROOT_PATH, 'y_' + str(i) + '.npy'), 'rb') as file:
                    y = joblib.load(file) 
                # X = np.random.shuffle(X)
                # y = np.ran
                for j in range( (len(X) // batch_size) - 1):
                    yield X[j * batch_size: (j+1) * batch_size], y[ j * batch_size: (j+1) * batch_size]  


def train():

    model = create_base_model(w = None, trainable = True)
    model = add_custom_layers(model)
    adam = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # base_model = create_base_model()
    # model = add_custom_layers(base_model)
    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    # train_history = model.fit_generator( x_gen(32, valid = False), validation_data = x_gen(32, valid = True),steps_per_epoch= 200, validation_steps = 14, epochs= 50, callbacks=[es,sv])

    train_history = model.fit_generator( generator(32), validation_data = generator(32, valid = True),  steps_per_epoch= 100, validation_steps = 84, epochs= 500, callbacks=[sv, LogCallback()])
# DATA_ROOT_PATH = 'data/npy'  
# with open(os.path.join(DATA_ROOT_PATH, 'X_11' + '.npy'), 'rb') as file:
#     X = joblib.load(file)
# with open(os.path.join(DATA_ROOT_PATH, 'y_11' + '.npy'), 'rb') as file:
#     y = joblib.load(file) 
# print(np.array(X).shape, np.array(y).shape)  
# exit()


train()