import numpy as np
import matplotlib.pyplot as plt

from skimage import color, restoration
#from skimage.transform import hough_circle, hough_circle_peaks, rescale
#from skimage.feature import canny
# from skimage.draw import circle_perimeter
import pandas as pd
from skimage.io import imread,imsave, imshow
from os import path, listdir, makedirs, fsdecode, getcwd
from math import sqrt, log


class TrainData:
    def __init__(self, name):
        self.filename = name

    def make_train(self, class_ID='Target'):
        train_labels = pd.read_csv(self.filename)
        train_labels = train_labels.astype('str')
        train_labels['Id'] = train_labels['Id'].apply(lambda x: x + '.png')
        train_labels[class_ID] = train_labels[class_ID].apply(lambda x: x.split(' '))
        train_labels[class_ID] = train_labels[class_ID].apply(lambda x: list(map(lambda x: int(x), x)))
        '''train_labels['Cat'] = None
        for index in range(len(train_labels[class_ID])):
            train_labels['Cat'][index] = [0] * 28
            for index_val in train_labels[class_ID][index]:
                train_labels['Cat'][index][index_val] = 1'''
        return train_labels

    def make_histdata(self, data):
        hist_data = {x: x * 0 for x in range(28)}
        for cat_index in range(28):
            for index in range(len(data['Target'])):
                hist_data[cat_index] += data['Cat'][index][cat_index]
        return hist_data

    def calc_weights(self, labels_dict, method, mu=0.15):
        if method == 'simple':
            total = np.sum(list(labels_dict.values()))
            keys = labels_dict.keys()
            class_weight = dict()

            for key in keys:
                score = log(mu * float(total) / float(labels_dict[key]))
                class_weight[key] = score if score > 1.0 else 1.0

            return class_weight
        if method == 'eurist':
            total = np.sum(list(labels_dict.values()))
            keys = labels_dict.keys()
            class_weight = dict()

            for key in keys:
                score = float(total) / (len(labels_dict) * float(labels_dict[key]))
                class_weight[key] = score if score > 1.0 else 1.0

            return class_weight





def save_rgb_image(img_id, classes, img_dir, labeled_dir, save_img=True):
    red_img_path = path.join(img_dir, img_id + '_red.png')
    green_img_path = path.join(img_dir, img_id + '_green.png')
    blue_img_path = path.join(img_dir, img_id + '_blue.png')
    yellow_img_path = path.join(img_dir, img_id + '_yellow.png')

    red_img = imread(red_img_path)
    green_img = imread(green_img_path)
    blue_img = imread(blue_img_path)
    yellow_img = imread(yellow_img_path)
    rgb_img = np.stack((red_img, green_img, blue_img), axis=2)
    red_mod = (rgb_img[:, :, 0]+yellow_img) * (255 / np.max((rgb_img[:, :, 0]+yellow_img)))
    green_mod = (rgb_img[:, :, 1]+yellow_img) * (255 / np.max((rgb_img[:, :, 1]+yellow_img)))
    rgby_img = np.stack((red_mod.astype(np.uint8), green_mod.astype(np.uint8), blue_img), axis=2)
    #imshow(rgb_img)
    #plt.show()
    #imshow(rgby_img)
    #plt.show()

    if save_img:
        for number, identifier in enumerate(classes):
            if identifier == 1:
                class_path = labeled_dir #+ '/'+str(number)
                if not path.exists(class_path):
                    makedirs(class_path)
                rgb_img_path = path.join(class_path, img_id + '.png')
                #rgby_img_path = path.join(class_path, img_id + 'y.png')
                imsave(rgb_img_path, rgb_img)
                #imsave(rgby_img_path, rgby_img)
    else:
        return rgb_img

#save_rgb_image('000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0', [1], train_dir)
if __name__ == '__main__':
    train_dir = getcwd() + '\\train'
    classes_dir = getcwd() + '\\new_train'
    if not path.exists(classes_dir):
        makedirs(classes_dir)

    train_df = TrainData('train.csv').make_train()
    i = 0

    for line in train_df.values:
        save_rgb_image(line[0], [1], train_dir, classes_dir)
        i += 1
        if (i % 100) == 0:
            print('{} images were saved'.format(i))





