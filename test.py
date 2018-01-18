from scipy.misc import imread, imsave, imresize
import os,csv
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
def boxing(img):
    # img_ori=mpimg.imread('data/bbox/'+imgID)
    # img=mpimg.imread('saliency_map/'+imgID)

    dimension = img.shape
    # dimension2 = img_ori.shape
    # print(dimension2)
    threshold = 0.95
    # for color in range(dimension[2]):
    #     for h in range(dimension[0]):
    #         for w in range(dimension[1]):
    #             # print(img[h][w][color])
    #             pass


    # 84 22
    # 500 22
    # 84 441
    # 500 441


    maxTop = maxLeft = 999999999
    maxRight = maxBottom = -1
    for h in range(224):
        for w in range(224):
            # print(h,w)
            if img[h][w][0] > threshold:
                if h < maxTop: maxTop = h
                if h > maxBottom: maxBottom = h
                if w < maxLeft: maxLeft = w
                if w > maxRight: maxRight = w


    maxTop = int((maxTop)/(224)*1024)
    maxBottom = int((maxBottom)/(224)*1024)
    maxLeft = int((maxLeft)/(224)*1024)
    maxRight = int((maxRight)/(224)*1024)


    box = {}
    box['x'] = maxLeft
    box['y'] = 1024 - maxBottom
    box['w'] = maxRight - maxLeft
    box['h'] = maxBottom - maxTop
    return box

LABEL_NAME = [
    'Atelectasis',
    'Cardiomegaly',
    'Effusion',
    'Infiltration',
    'Mass',
    'Nodule',
    'Pneumonia',
    'Pneumothorax'
]

# load model
model = load_model('resnet_no_weight_model.h5')
b_model = load_model('bin_model.h5')

layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == "dense_2"][0]
layer_idx_b = [idx for idx, layer in enumerate(b_model.layers) if layer.name == "dense_2"][0]


def test():
    filenames = []
    with open(VALID_DATA_PATH, 'r') as f:
        re = csv.reader(f)
        for r in re:
            filenames.append(r[0])

    for img in filenames:
        img_data = imresize(imread('data/images/'+ img, mode ='RGB') ,size=(224,224))
   
        X_test = np.array([img_data])
        result = model.predict(X_test)
        result_classes = result.argmax(axis=-1)
        result_classes = list(result_classes)
        # print(result)
        result2 = b_model.predict(X_test)
        for x in result2:
            if  x[0] > 0.65:
                result_classes.append(3) # 4-1
            else:
                pass

        stupidModelList = [model, b_model, model] # 3rd should never be used
        stupidLayerIdxList = [layer_idx, layer_idx_b, layer_idx]

        with open ("result.txt", "w") as outputfile:   
            outputfile.write((img+ ' '+str(len(result_classes)) + '\n')) # result_classes must within 1~2
            
            for idx, x in enumerate(result_classes):
                heatmap = visualize_saliency(stupidModelList[idx], stupidLayerIdxList[idx], np.expand_dims(result_classes[idx], axis=0), img_data)

                box = boxing(heatmap)
                print(box)
                print(LABEL_NAME[result_classes[idx]])
                outputfile.write(('%s %f %f %f %f\n' % (LABEL_NAME[result_classes[idx]], box['x'], box['y'], box['w'], box['h'])))

                del heatmap
                exit()

    

        # for idx, x in enumerate(X_valid):
        #     heatmap = visualize_saliency(model, layer_idx, np.expand_dims(result_classes[idx], axis=0), x)
        #     heatmap = None
        #     del heatmap
        #     exit()
        
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
   
VALID_DATA_PATH = sys.argv[1]
test()