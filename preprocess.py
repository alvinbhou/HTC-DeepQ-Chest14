import csv
import pickle
 
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
    'Infiltration': -1,
    'Infiltrate': 4,
    'Mass': 5,
    'Nodule': 6,
    'Pleural_Thickening': -1,
    'Pneumonia': 7,
    'Pneumothorax': 8
}
 
testset = set()
validset = set()
trainset = set()
 
with open('data/train.txt') as f:
    re = csv.reader(f)
    for r in re:
        trainset.add(r[0])
 
with open('data/valid.txt') as f:
    re = csv.reader(f)
    for r in re:
        validset.add(r[0])
 
with open('data/test.txt') as f:
    re = csv.reader(f)
    for r in re:
        testset.add(r[0])
 
traindata = {}
validdata = {}
testdata = {}
 
with open('data/Data_Entry_2017_v2.csv') as f:
    re = csv.reader(f)
    next(re)
    for r in re:
        id = r[0]
        obs = None
        for observe in r[1].split('|'):
            ob = LABELS[observe]
            if ob == -1:
                continue
            obs = ob-1
            break
 
        if obs == None:
            continue
 
        if id in trainset:
            traindata[id] = obs
        elif id in validset:
            validdata[id] = obs
        elif id in testset:
            testdata[id] = obs
 
print(len(traindata.keys()))
print(len(validdata.keys()))
print(len(testdata.keys()))
with open('data/pickles/labels_train.pkl', 'wb') as f:
    pickle.dump(traindata, f)
 
with open('data/pickles/labels_valid.pkl', 'wb') as f:
    pickle.dump(validdata, f)
 
with open('data/pickles/labels_test.pkl', 'wb') as f:
    pickle.dump(testdata, f)