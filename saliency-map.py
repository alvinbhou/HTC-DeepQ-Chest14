import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
from scipy.misc import imread, imsave, imresize
import csv
import tensorflow as tf 
from keras.models import load_model
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input
from vis.utils import utils
from vis.visualization import visualize_saliency


def loadData():
    X = []
    Y = []
    f = open('train.csv')  
    num = 0  
    for row in csv.reader(f):       
        if(row[0] == 'label'):
            continue       
      
        x = np.array(row[1].split(" "))
        x = np.array(list(map(float, x))).reshape((48,48,1))
        X.append(x)
       
        num = num + 1
    X = np.array(X)
    X = X / 255 
    # Y = keras.utils.to_categorical(Y,  num_classes = 7)
    return X



model_path = "./model/new3_model.h5"
model = load_model(model_path)

layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == "dense_3"][0]


CLASS_NAME = "Cardiomegaly"
CLASS_ID = 1
for (dirpath, dirnames, filenames) in os.walk('data/train/' + CLASS_NAME):
    for filename in filenames:
        img = imread( os.path.join('data/train', CLASS_NAME, filename), mode ='RGB')
        img = img / 255
        img = imresize(img ,size=(224,224))
        IMG_ID = filename[:-4]
        print(IMG_ID)




	
        val_proba = model.predict(np.expand_dims(img, axis=0))
        pred = val_proba.argmax(axis=-1)

        heatmap = visualize_saliency(model, layer_idx, np.expand_dims(CLASS_ID, axis=0), img)

        threshold = 0.35
        see = img
        mp = np.mean(see)
        for i in range(0,224):
            for j in range(0,224):
                if np.mean(heatmap[i][j]) / 255 < threshold:
                    see[i][j] = mp

        if not os.path.exists("saliency_map/" + CLASS_NAME):
            os.makedirs("saliency_map/" +CLASS_NAME)

        plt.figure()
        plt.imshow(heatmap, cmap=plt.cm.jet)
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig( os.path.join("saliency_map/" ,CLASS_NAME , IMG_ID +".png"), dpi=100)
        plt.close(fig)
        
        plt.figure()
        plt.imshow(see,cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join("saliency_map/" ,CLASS_NAME , IMG_ID +"_mask.png"), dpi=100)
        plt.close(fig)
        plt.close('all')
        