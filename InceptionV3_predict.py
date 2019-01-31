from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import InceptionV3_fine_tune2
import img_proc_scikit
from PIL import Image, ImageFile
from os import path, listdir, makedirs, fsdecode, getcwd
from skimage.transform import resize

ImageFile.LOAD_TRUNCATED_IMAGES = True


train_df = img_proc_scikit.TrainData('train.csv').make_train()
img_rows, img_cols = 299, 299  # Resolution of inputs
channel = 3
num_classes = 28
batch_size = 12

model = InceptionV3_fine_tune2.InceptionV3_make('InceptionV3_ft2.h5', 28)
f1_score = InceptionV3_fine_tune2.f1

train_dir = getcwd() + '\\train'
train_df['Preds'] = None
i = 0

for line in train_df.values:
    sample = img_proc_scikit.save_rgb_image(line[0], line[2], train_dir, classes_dir=None, save_img=False)
    sample = resize(sample, (299, 299))
    sample = np.array(sample)
    line[3] = model.predict(np.expand_dims(sample, axis=0), batch_size=1)[0]
    i += 1
    if (i % 100) == 0:
        print('{} images were predicted'.format(i))







