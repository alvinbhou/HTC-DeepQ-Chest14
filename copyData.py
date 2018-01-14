import  shutil ,os, pickle
LABELS = [
    'Atelectasis',
    'Cardiomegaly',
    'Effusion',
    'Infiltrate',
    'Mass',
    'Nodule',
    'Pneumonia',
    'Pneumothorax'
]
# for d in LABELS:
#     os.makedirs(os.path.join('data', 'train', d))
#     os.makedirs(os.path.join('data', 'validation', d))
#     os.makedirs(os.path.join('data', 'test', d)) 

from random import shuffle

with open('data/pickles/labels_valid.pkl', 'rb') as f:
    traindata = pickle.load(f)

for k, v in traindata.items(): 
    print(k)
    src = 'data/images/' + k
    dest = os.path.join('data', 'validation', LABELS[v], k)
    shutil.copyfile(src, dest)
# with open('data/pickles/labels_train.pkl', 'rb') as f:
#     traindata = pickle.load(f)

# for k, v in traindata.items(): 
#     print(k)
#     src = 'data/images/' + k
#     dest = os.path.join('data', 'train', LABELS[v], k)
#     shutil.copyfile(src, dest)
        