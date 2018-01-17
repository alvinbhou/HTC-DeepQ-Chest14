import os, pickle
from sklearn.externals import joblib
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

with open('data/npy/X_valid.npy', 'rb') as f:
    X = joblib.load(f)
with open('data/npy/y_valid.npy', 'rb') as f:
    y = joblib.load(f)

with open('data/pickles/labels_valid.pkl', 'rb') as f:
    valid = pickle.load(f)
    for k, v in valid.items():
        print(k, v)
        break

print(X[0])
print(y[0])