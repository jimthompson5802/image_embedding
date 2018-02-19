#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 06:45:12 2018

Demonstration of visualization of image embeddings using tensorboard

@author: jim
"""
#%%
import tensorflow as tf
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.utils import to_categorical
# fix dimension ordering issue
from tensorflow.python.keras import backend as K
print(K.image_data_format())

from tensorflow.contrib.tensorboard.plugins import projector

import matplotlib.pyplot as plt
import os.path
import os
import numpy as np

from skimage import io as imgio, transform, util as imgutil

#%%
# location for tensorboard data structures
LOG_DIR = './cat_dog_embeddings_logs'

# location of image files
IMG_DIR = '/Users/jim/Desktop/Deeplearning/transfer_learning/cats_dogs/data/train'

# number of cat and dog images to select
IMGS_TO_SELECT = 100

#%%
# =============================================================================
# select image files to create embedding
# =============================================================================
cat_imgs = np.array(os.listdir(os.path.join(IMG_DIR,'cats')))
dog_imgs = np.array(os.listdir(os.path.join(IMG_DIR,'dogs')))

# set seed
np.random.seed(13)

cats = cat_imgs[np.random.choice(cat_imgs.shape[0],IMGS_TO_SELECT,replace=False)]
dogs = dog_imgs[np.random.choice(dog_imgs.shape[0],IMGS_TO_SELECT,replace=False)]




#%%
###
# Function to get the largeest centered square image
###
def center_crop_img(img):
    """CNN's expect square images.
    Take the largest possible square out of the middle.
    """
    # First, find center square crop
    wid,hgt = img.shape[:2]
    if( wid > hgt ):
        clip =  hgt // 2
        beforeh=0
        afterh=hgt
        beforew = wid//2 - clip
        afterw = wid//2 + clip
    else:
        clip = wid // 2
        beforeh = hgt//2 - clip
        afterh = hgt//2 + clip
        beforew=0
        afterw=wid
        
    img = img[beforew:afterw,beforeh:afterh,:]
    
    return img

#%%
# =============================================================================
# Retrieve pre-trained image CNN to generate embeddings
# =============================================================================

# retrireve VGG16 CNN
from tensorflow.python.keras.applications import VGG16
vgg16 = VGG16()
vgg16.summary()


#%%
# remove final layer
vgg16.layers.pop()
vgg16.summary()

#%%
vgg16.compile(optimizer='Adam',loss='mean_squared_error')

#%%
def generate_embedding(img):
    img = center_crop_img(img)
    img = transform.resize(img,(224,224)).reshape(-1,224,224,3)
    embedding = vgg16.predict(img)
    return embedding
    
#%%
img = imgio.imread(os.path.join(IMG_DIR,'cats',cat_imgs[0]))
plt.imshow(img);plt.show()

img2 = center_crop_img(img)
plt.imshow(img2);plt.show()


img3 = transform.resize(img2,(224,224))
plt.imshow(img3);plt.show()

#%%
vgg16.summary()


#%%
embed = generate_embedding(img)