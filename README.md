
### 實驗環境
OS: 	Ubuntu 16.04.2 LTS


Python Version: 3.6.0

#### 用到的pacakge與版本
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

####  檔案資料夾結構

檔案的資料結構擺放方式如下
```
.
data/
        BBox_List_2017.csv
        Data_Entry_2017_v2.csv
        train.txt
        valid.txt
        test.txt
        images/
            ..(42G圖片於此)
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
... (其他python檔案)
```

### 如何跑 training 
1. 先做資料預處理，執行
`./preprocess.sh `
會將資料預處理存於`data/npy`和`data/bin`之下，可能約須一個多小時的執行時間
2. 接著進行訓練，執行
`./train.sh` 會分別訓練兩個不同的model (8類分類器 + classfier) 於`models/classifier_model` 和 `models/binary_model` 下
### 如何跑 testing 
如使用hTc的judge平台
1. 先下載model和必要python檔案
執行 `./download.sh ` 下載 `test.zip`

2. `test.zip`內有 `py_new.py`, `judge_runner.py`, `resnet_no_weight_model.h5`, `requirements3.txt`, `bin_model.h5` 五個檔案。將其壓縮檔上傳於judge並以 `python3 judge_runner.py` 的command執行。




