from keras import Model
from keras.applications import InceptionV3
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D
import keras
from keras.utils import np_utils
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.model_selection import train_test_split
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import csv
from pathlib import Path
import urllib.request
from PIL import Image
import time
import random
from keras.callbacks import Callback

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target, Image.ANTIALIAS)
    image = np.asarray(image)
    X = np.empty((1, 299, 299, 3))
    X[0] = image
    # return the processed image
    return X



from keras.models import load_model
np.set_printoptions(precision=4)

img = Image.open('../documents/Residency/FF959AE4E89B488195844E1D1C2893A0.jpg')
X = prepare_image(img,(299,299))
print(X)
model = load_model('snapshot/vgg19.2018-08-21 21:00.9-03-0.19-0.95.h5')
print([round(x,4) for x in  model.predict(X)[0]])

