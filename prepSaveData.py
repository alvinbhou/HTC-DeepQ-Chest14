from scipy.misc import imread, imsave, imresize
import pickle
import numpy as np
import os
import cv2
import sys, random, time
from sklearn.externals import joblib
from keras.utils import to_categorical

CLASSES = 8
with open('data/pickles/labels_train.pkl', 'rb') as f:
    traindata = pickle.load(f)
X, y = [], []
count = 0
for k, v in traindata.items(): 

    print(k , v)
    exit(1)      
    img = imread('data/images/' + k, mode ='RGB')
    img = img / 255
    X.append(imresize(img ,size=(224,224)))
    y.append(v)
    if(len(X) == 3200):
        print(count)
        y = to_categorical(y, num_classes=CLASSES)
        X = np.array(X)
        with open('data/npy/X_' + str(count) +'.npy', 'wb') as f:
            joblib.dump(X, f)
        with open('data/npy/y_' + str(count) +'.npy', 'wb') as f:
            joblib.dump(y, f)
        X, y = [], []
        count += 1
count += 1  
print(len(X))     
with open('data/npy/X_' + str(count) +'.npy', 'wb') as f:
    joblib.dump(X, f)
with open('data/npy/y_' + str(count) +'.npy', 'wb') as f:
    joblib.dump(y, f)
