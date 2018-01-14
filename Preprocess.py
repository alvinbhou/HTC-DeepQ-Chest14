
# coding: utf-8

# In[1]:


import cv2
import numpy as np


# In[2]:


IMG_PATH = 'data/2.png'


# In[7]:


def EqualizeHistogram(img, in_path=None):
    if in_path != None:
        img = cv2.imread(in_path,0)
    equ = np.array(cv2.equalizeHist(img))
    print(equ.shape)
    return equ

#     res = np.hstack((img,equ)) #stacking images side-by-side
    # cv2.imwrite(out_path,equ)


# In[4]:


def CLAHE(img, in_path = None, tileGridsize=(8,8)):
    if in_path != None:
        img = cv2.imread(in_path,0)
    clahe1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=tileGridsize)
    cl1 = clahe1.apply(img)
    return cl1
img = None
img = EqualizeHistogram(img, in_path = './data/images/00004547_009.png')
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
img = CLAHE(img, in_path = './data/images/00004547_009.png')
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


    
def Median(in_path, out_path, k_size=5):
    img = cv2.imread(in_path,0)
    median = cv2.medianBlur(img, k_size)
#     res = np.hstack((img,median)) #stacking images side-by-side
    cv2.imwrite(out_path, median)
    
def Closing(in_path, out_path, k_size=(5,5)):
    kernel = np.ones(k_size, np.uint8)
    img = cv2.imread(in_path,0)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(out_path, closing)
    
def Opening(in_path, out_path, k_size=(5,5)):
    kernel = np.ones(k_size, np.uint8)
    img = cv2.imread(in_path,0)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(out_path, opening)


# In[8]:



# CLAHE(IMG_PATH, 'ttttt.png')
# Median(IMG_PATH, 'MMM.png')
# Opening(IMG_PATH,'OOOOO.png')
# Closing(IMG_PATH,'CCCCC.png')


# In[6]:


# clahe1 = cv2.createEEEE.pngCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(15,15))
# clahe3 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(25,25))
# cl1 = clahe1.apply(img)
# cl2 = clahe2.apply(img)
# cl3 = clahe3.apply(img)
# cv2.imwrite('clahe_tilesize.jpg',np.hstack((cl1,cl2,cl3)))

# median = cv2.medianBlur(cl1,5)
# cv2.imwrite('median.jpg',np.hstack((cl1,median)))

# img2 = cv2.imread('data/3.png',0)
# clahe_ = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10,10))
# cl_ = clahe_.apply(img2)

# kernel = np.ones((5,5),np.uint8)
# closing = cv2.morphologyEx(cl_, cv2.MORPH_CLOSE, kernel)
# opening = cv2.morphologyEx(cl_, cv2.MORPH_OPEN, kernel)
# cv2.imwrite('closing.jpg',np.hstack((cl_,closing,opening)))

