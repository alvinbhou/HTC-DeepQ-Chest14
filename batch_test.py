import judger_medical as judger
import numpy as np
import csv, os
from keras.models import Model, load_model
from vis.visualization import visualize_saliency
from scipy.misc import imread, imsave, imresize

# import matplotlib.pyplot as plt



# input saliency
def boxing(img):
    # img_ori=mpimg.imread('data/bbox/'+imgID)
    # img=mpimg.imread('saliency_map/'+imgID)

    dimension = img.shape
    # dimension2 = img_ori.shape
    # print(dimension2)
    threshold = 0.90
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




imgs = []
with open("filenames.txt","r") as filenames:
    re = csv.reader(filenames)
    for r in re:
        imgs.append(r[0])


with open ("result.txt", "w") as outputfile: 
    for img in imgs:
        print(img)
        img_data = imresize(imread(os.path.join(img), mode ='RGB') ,size=(224,224))
        X_test = np.array([img_data])
        result = model.predict(X_test)
        result_classes = list(result.argmax(axis=-1))
        # print(result)
        result2 = b_model.predict(X_test)
        for x in result2:
            if  x[0] > 0.3:
                result_classes.append(3) # 4-1
            else:
                pass
                # result_classes.append(0)

        stupidModelList = [model, model, model] # 3rd should never be used
        stupidLayerIdxList = [layer_idx, layer_idx, layer_idx]

        flag = True
        outputfile.write((img+ ' '+str(len(result_classes)) + '\n')) # result_classes must within 1~2
        
        for idx, x in enumerate(result_classes):
            if(flag):
                heatmap = visualize_saliency(stupidModelList[idx], stupidLayerIdxList[idx], np.expand_dims(result_classes[idx], axis=0), img_data)
                flag = False
            box = boxing(heatmap)
            print(box)
            print(LABEL_NAME[result_classes[idx]])
            outputfile.write(('%s %f %f %f %f\n' % (LABEL_NAME[result_classes[idx]], box['x'], box['y'], box['w'], box['h'])))

            # del heatmap

if os.path.exists("filenames.txt"):
    os.remove("filenames.txt")
    
