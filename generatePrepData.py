from scipy.misc import imread, imsave, imresize
import pickle
import numpy as np
import os
import cv2, csv
import sys, random, time
from sklearn.externals import joblib
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from keras.utils import to_categorical
CLASSES = 8
 
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


# binary data
def bin_preprocess():
    trainset = set()
    with open('data/train.txt') as f:
        re = csv.reader(f)
        for r in re:
            trainset.add(r[0])

    isInfiltration = []
    notInfiltration = []
    with open('data/Data_Entry_2017_v2.csv') as f:
        re = csv.reader(f)
        next(re)
        for r in re:
            id = r[0]
            flag = False
            for observe in r[1].split('|'):
                ob = LABELS[observe]
                if observe == 'Infiltration':
                    isInfiltration.append(id)
                    flag = False
                    break
                if ob != -1:
                    flag = True
        
            if (flag):
                notInfiltration.append(id)
    print(len(isInfiltration), len(notInfiltration))

    X = []
    y = []

    for idx in range(1,10001):
        img1 = imread('data/images/' + isInfiltration[idx], mode ='RGB')
        img2 = imread('data/images/' + notInfiltration[idx], mode ='RGB')
        X.append(imresize(img1 ,size=(224,224)))
        y.append([1])
        X.append(imresize(img2 ,size=(224,224)))
        y.append([0])
        # if(idx % 10 == 0):
        #     print(idx)
        if(idx % 2000 == 0):
            print(idx)
            X = np.array(X)
            y = np.array(y)  
            with open('data/bin/X_' + str(idx // 2000) +'.npy', 'wb') as f:
                joblib.dump(X, f)
            with open('data/bin/y_' + str(idx // 2000) +'.npy', 'wb') as f:
                joblib.dump(y, f)
            X = []
            y = []
if not (os.path.exists("data/bin/")):
    os.makedirs("data/bin/")
if not (os.path.exists("data/npy")):
    os.makedirs("data/npy")
bin_preprocess()


# valid
with open('data/pickles/labels_valid.pkl', 'rb') as f:
    validdata = pickle.load(f)
X, y = [], []
for k, v in validdata.items(): 
    img = imread('data/images/' + k, mode ='RGB')
    # img = cv2.imread('data/images/' + k,0)
    # resized_image = cv2.resize(img, (224, 224)) 
    # img = EqualizeHistogram(resized_image)
    # img = CLAHE(img)
    # img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    X.append(imresize(img ,size=(224,224)))
    # X.append(img)
    
    y.append(v)
          
y = to_categorical(y, num_classes=CLASSES)
X = np.array(X)
print(X.shape, y.shape)
with open('data/npy/X_valid.npy', 'wb') as f:
    joblib.dump(X, f)
with open('data/npy/y_valid.npy', 'wb') as f:
    joblib.dump(y, f)



# train
with open('data/pickles/labels_train.pkl', 'rb') as f:
    traindata = pickle.load(f)
X, y = [], []
count = 0
item_count = 0
for k, v in traindata.items(): 
    # if(item_count % 10 == 0):
    #     print(item_count)
    # img = cv2.imread('data/images/' + k,0)
    # resized_image = cv2.resize(img, (224, 224)) 
    # img = EqualizeHistogram(resized_image)
    # img = CLAHE(img)
    # img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    img = imread('data/images/' + k, mode ='RGB')
    X.append(imresize(img ,size=(224,224)))
    y.append(v)
    if(len(X) == 1600):
        print(count)
        y = to_categorical(y, num_classes=CLASSES)
        X = np.array(X)
        with open('data/npy/X_' + str(count) +'.npy', 'wb') as f:
            joblib.dump(X, f)
        with open('data/npy/y_' + str(count) +'.npy', 'wb') as f:
            joblib.dump(y, f)
        X, y = [], []
        count += 1
    item_count += 1
    if(count >= 20):
        break
count += 1  
print(len(X))   
X = np.array(X)
y = to_categorical(y, num_classes=CLASSES)
with open('data/npy/X_' + str(count) +'.npy', 'wb') as f:
    joblib.dump(X, f)
with open('data/npy/y_' + str(count) +'.npy', 'wb') as f:
    joblib.dump(y, f)

