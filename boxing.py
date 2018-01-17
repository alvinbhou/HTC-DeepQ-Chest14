import csv
import pickle
import  shutil
from scipy.misc import imread, imsave, imresize

files = []
with open('data/BBox_List_2017.csv') as f:
    re = csv.reader(f)
    next(re)
    for r in re:
        shutil.copyfile('data/images/' + r[0], 'data/boxing_data/' + r[0])
        files.append(r[0])
print(len(files))