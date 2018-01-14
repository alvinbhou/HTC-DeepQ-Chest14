from scipy.misc import imread, imsave, imresize
from keras.utils import to_categorical
from keras.layers import Conv2D, BatchNormalization, Input, Activation
from keras.layers import LeakyReLU, Lambda, Reshape, Concatenate, Add, Dense, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.optimizers import SGD, Adam
from keras import regularizers
import keras.backend as K
from keras.models import Model
import pickle
import numpy as np
import os
import cv2
from collections import Counter
import sys, random, time
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.resnet50 import ResNet50, preprocess_input
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
                            save_weights_only=True,
                            mode='auto',
                            period=1)
es = EarlyStopping(patience=7)


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

def create_base_model():
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in model.layers:
        layer.trainable = False
    return model

def rebase_base_model(model):
    # for layer in model.layers[:self.num_fixed_layers]:
    #     layer.trainable = False
    for layer in model.layers:
        layer.trainable = True
    return model

def create_data_generator():
    train_image_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.05,
        zoom_range=0.1,
        fill_mode='nearest',
        horizontal_flip=False,
        vertical_flip=False
    )
    test_image_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

    return train_image_gen, test_image_gen
def add_custom_layers(base_model):
        x = base_model.output
        x = Flatten()(x)
        x = Dense(512, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.005))(x)
        x = Dense(512, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.005))(x)
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
    x = Flatten()(x)
    x = Dense(512, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.005))(x)
    x = Dense(512, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.005))(x)
    p = Dense(512,  kernel_initializer='glorot_normal', activation='relu')(ir.output)
    p = Dense(512,  kernel_initializer='glorot_normal', activation='relu')(ir.output)
    p = Dense(CLASSES,activation='softmax')( p)

    # ============ Model
    model = Model(ir.input, p)
    model.compile('adam', 'categorical_crossentropy', metrics= ['acc'])
    # model.summary()
    return model

def create_class_weights( y, smooth_factor=0.2):
    counter = Counter(y)
    if smooth_factor > 0:
        p = max(counter.values()) * smooth_factor
        for k in counter.keys():
            counter[k] += p
    majority = max(counter.values())
    return {cls: float(majority / count) for cls, count in counter.items()}

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
    # model = create_model_ResNetV2()
    base_model = create_base_model()
    model = add_custom_layers(base_model)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    train_image_gen, test_image_gen = create_data_generator()

    train_gen = train_image_gen.flow_from_directory('./data/train', target_size=(224, 224), batch_size=64)
    weights = create_class_weights(train_gen.classes, 0.15)
    validation_gen = test_image_gen.flow_from_directory('./data/validation', target_size=(224, 224), batch_size=64)
    model.fit_generator(
            train_gen,
            epochs=30,
            steps_per_epoch= 320,
            validation_data=validation_gen,
            validation_steps= 440//64 + 1,
            class_weight=weights,
            callbacks=[es, sv]
        )

    # second run
    model = rebase_base_model(model)
    adam = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    MODELFILE = 'model2.h5'
    sv2 = ModelCheckpoint(os.path.join(MODELDIR, MODELFILE),
                            monitor='val_acc',
                            verbose=1,
                            save_best_only=True,
                            save_weights_only=True,
                            mode='auto',
                            period=1)
    model.fit_generator(
            train_gen,
            epochs=30,
            steps_per_epoch= 320,
            validation_data=validation_gen,
            validation_steps= 440//64 + 1,
            class_weight=weights,
            callbacks=[es, sv2]
        )
    
    
train()
     