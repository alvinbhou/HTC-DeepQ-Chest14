# Weakly Supervised Learning for Findings Detection in Medical Image
## What is the project about?
This project aims to classify diseases and identifying abnormalities based on medical X-ray images (Chest X-ray 14 dataset). Competed in the HTC DeepQ DL competitions at NTU.

## Setup 
### Environment
Ubuntu 16.04 LTS

### Required pacakges
```
Keras==2.0.9
keras-vis==0.4.1
opencv-python==3.3.0.10
pandas==0.20.1
pydot==1.2.3
matplotlib==2.0.2
scipy==0.19.0
tensorflow-gpu==1.3.0
```
### Directory tree
```
.
data/
        BBox_List_2017.csv
        Data_Entry_2017_v2.csv
        train.txt
        valid.txt
        test.txt
        images/
            ..(42G Images here)
        npy/
            X_0.npy
            y_0.npy
            X_1.npy
            … (Preprocessed data for classfier)
        bin/
            X_1.npy
            y_1.npy
            X_2.npy
            … (Preprocessed data for binary classfier)
preprocess.sh 
download.sh
train.sh 
train.py
binary.py
preprocessLabel.py
generatePrepData.py
test.py
dropbox.py
... (other python files)
```
### Training 
1. Data preprocessing, run `./preprocess.sh ` and the processed data will be saved at `data/npy` and `data/bin`

2. To start training, run `./train.sh` will start to train two different models (8+1 classifiers) and the models will be saved at `models/classifier_model` 和 `models/binary_model` 

### Testing 
How to run testing on the HTC DeepQ Platform

1. To download models and the required python-files, run `./download.sh ` to download `test.zip`

2. Inside `test.zip` includes `batch_test.py`, `judge_runner.py`, `resnet_no_weight_model.h5`, `requirements3.txt`, `bin_model.h5` 5 files. Upload the zip file to the judge and run the command `python3 judge_runner.py` to start the testing process.

## Methods and Models
### Data preprocessing
#### Histograms Equalization
Since all the X-rat images are black and white, the distribution of the Histogram is quite important. We applied histogram equalization to the original dataset's image to standardize it.

Left: Original image, Right: Applied histogram equalization

<img src="https://i.imgur.com/HV0bF1Q.png" height="300px">

#### CLAHE (Contrast Limited Adaptive Histogram Equalization)
The drawbacks of Histograms Equalization is that it uses the global contract of images as a basis for standardization. This may result in some "over-brightness" on important areas in the image. From the image above, we found that the right image's rib is not clear enough. So we decided to use CLAHE to limit the contract difference to prevend over brightness.

Left: Applied histogram equalization, Right: Applied histogram equalization and CLAHE

<img src="https://i.imgur.com/C2w4K1S.png" height="300px">

### Training methods
We used ResNet-50 and InceptionResNetV2 to differenet models for our training.

![](https://i.imgur.com/ePGul0R.png)

(Picture Reference: http://book.paddle.org/03.image_classification/)

The first step is to train a well performing classifier and use the model's last layer's saliency map to find out where the network is focusing on. Moreover, we can generate bounding boxed to identify abnormalities by filtering through the generated heatmap.

![](https://i.imgur.com/wJWpsJb.png)

This is a saliency map of a Cardiomegaly patient, and we can tell it focuses on the left heart.

## Performances
Using the ImageNet pre-trained weights with frozen layers, the validation accuracy is around 40%.

<img src="https://i.imgur.com/oPbvZc3.png" height="250px">

## Bounding results
### Use Effusion as an example
Black: ground truth, Yellow: predicted
![](https://i.imgur.com/qVWHNOr.png)

![](https://i.imgur.com/Gbo4fQb.png)

The boxing method is to bound the area whrere its red pixel value is greater than a threshold. The problem is that sometimes outliers will make the predict area too large and inaccurate. We have tried averaging methods to remove outliers, but did not seem to improve much.

## Conclusion and Further Research
The training results is not spectacular, with a 41% accuracy. It may result from bad model selection on our end. However, we done some further research and found that Luke Oakden-Rayner, a PhD Candidate and Radiologist, has some thoughts on this dataset

```
 "I believe the ChestXray14 dataset, as it exists now, is not fit for training medical AI systems to do diagnostic work."
```

He point out that there are some major issues
* how accurate the labels are
* what the labels actually mean, medically
* how useful the labels are for image analysis

which leads to bad performance in machine learning tasks. Maybe we could try more to improve in the future.

## Reference
[Exploring the ChestXray14 dataset: problems](https://lukeoakdenrayner.wordpress.com/2017/12/18/the-chestxray14-dataset-problems/)

[ Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

[ Classification of Common Thorax Diseases ](https://github.com/srm-soumya/chest-scan-classifier)
