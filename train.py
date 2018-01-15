from scipy.misc import imread, imsave, imresize
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from keras.utils import to_categorical
from keras.layers import Conv2D, BatchNormalization, Input, Activation
from keras.layers import LeakyReLU, Lambda, Reshape, Concatenate, Add, Dense, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler, Callback
from keras.optimizers import SGD, Adam
from keras import regularizers
import keras.backend as K
from keras.models import Model
import pickle
import numpy as np
from sklearn.externals import joblib
import cv2
import sys, random, time
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.resnet50 import ResNet50, preprocess_input

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
CLASSES =  8 

# sv = ModelCheckpoint(os.path.join(MODELDIR, MODELFILE), save_best_only=True, save_weights_only=True)
sv = ModelCheckpoint(os.path.join(MODELDIR, MODELFILE),
                            monitor='val_acc',
                            verbose=1,
                            save_best_only=True,
                            save_weights_only=False,
                            mode='auto',
                            period=1)
es = EarlyStopping(patience=15)


def EqualizeHistogram(img, in_path=None):
    if in_path != None:
        img = cv2.imread(in_path,0)
    equ = np.array(cv2.equalizeHist(img))
    # print(equ.shape)
    return equ

def CLAHE(img, in_path = None, tileGridsize=(8,8)):
    if in_path != None:
        img = cv2.imread(in_path,0)
    clahe1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=tileGridsize)
    cl1 = clahe1.apply(img)
    return cl1

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
        x = Dense(2048, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.005))(x)
        x = Dense(2048, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.005))(x)
        y = Dense(CLASSES, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=y)
        return model

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

    
    p = Dense(512,  kernel_initializer='glorot_normal', activation='relu')(ir.output)
    p = Dense(1024,  kernel_initializer='glorot_normal', activation='relu')(ir.output)
    p = Dense(CLASSES,activation='softmax')(p)

    # ============ Model
    model = Model(ir.input, p)
    model.compile('adam', 'categorical_crossentropy', metrics= ['acc'])
    # model.summary()
    return model

def generator(batch_size, valid = False):
    DATA_ROOT_PATH = 'data/npy'
    while True:
        if(valid):
            with open(os.path.join(DATA_ROOT_PATH, 'X_11' + '.npy'), 'rb') as file:
                X = joblib.load(file)
            with open(os.path.join(DATA_ROOT_PATH, 'y_11' + '.npy'), 'rb') as file:
                y = joblib.load(file) 
            X = np.array(X)
            y = to_categorical(y, num_classes=CLASSES)
            for j in range((len(X) // batch_size) - 1):
                    yield X[j * batch_size: (j+1) * batch_size], y[ j * batch_size: (j+1) * batch_size]

        else:
            for i in range(0,10):
                with open(os.path.join(DATA_ROOT_PATH, 'X_' + str(i) + '.npy'), 'rb') as file:
                    X = joblib.load(file)
                with open(os.path.join(DATA_ROOT_PATH, 'y_' + str(i) + '.npy'), 'rb') as file:
                    y = joblib.load(file) 
                for j in range( (len(X) // batch_size) - 1):
                    yield X[j * batch_size: (j+1) * batch_size], y[ j * batch_size: (j+1) * batch_size]  

def x_gen(batch_size, valid = False):
    if(valid):
        with open('data/pickles/labels_valid.pkl', 'rb') as f:
            traindata = pickle.load(f)
    else:
        with open('data/pickles/labels_train.pkl', 'rb') as f:
            traindata = pickle.load(f)
    X, y = [], []
    while True:
        for k, v in traindata.items():
            # img = cv2.imread('data/images/' + k,0)
            # resized_image = cv2.resize(img, (256, 256)) 
            # img = EqualizeHistogram(resized_image)
            # img = CLAHE(img)
            # img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
            img = imread('data/images/' + k, mode ='RGB')
            img = img / 255
            X.append(imresize(img ,size=(224,224)))
            y.append(v[0])
            if(len(X) == batch_size):
                y = to_categorical(y, num_classes=CLASSES)
                X = np.array(X)
                yield X,y
                X, y = [], []

def train():
    model = create_base_model(w = None, trainable = True)
    model = add_custom_layers(model)
    adam = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # base_model = create_base_model()
    # model = add_custom_layers(base_model)
    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    # train_history = model.fit_generator( x_gen(32, valid = False), validation_data = x_gen(32, valid = True),steps_per_epoch= 200, validation_steps = 14, epochs= 50, callbacks=[es,sv])

    train_history = model.fit_generator( generator(64), validation_data = generator(64, valid = True),  steps_per_epoch= 100, validation_steps = 84, epochs= 100, callbacks=[es,sv, LogCallback()])
# DATA_ROOT_PATH = 'data/npy'  
# with open(os.path.join(DATA_ROOT_PATH, 'X_11' + '.npy'), 'rb') as file:
#     X = joblib.load(file)
# with open(os.path.join(DATA_ROOT_PATH, 'y_11' + '.npy'), 'rb') as file:
#     y = joblib.load(file) 
# print(np.array(X).shape, np.array(y).shape)  
# exit()
train()
     