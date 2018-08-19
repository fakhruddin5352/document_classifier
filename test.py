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


from keras.models import load_model

X = np.empty((1, 299, 299, 3))

img = Image.open("D:\\test\\b-3884.jpg").resize((299, 299), Image.ANTIALIAS)
arr = np.asarray(img)
X[0] = arr
model = load_model('inceptionv3-transfer-learning.model')
model.evaluate(X,np.asarray(range(12)),batch_size=1)

