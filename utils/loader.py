'''
    ------------------------------------
    Author : SAHLI Mohammed
    Date   : 2019-11-09
    Company: Alphaya (www.alphaya.com)
    Email  : nihon.sahli@gmail.com
    ------------------------------------
'''

import os
import cv2
import glob
import sklearn
import numpy as np

# -----------------------------------------------------------------------------------
def load_image(filename, width, height):
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height),0,0, cv2.INTER_LINEAR)
    image = np.multiply(image.astype(np.float32), 1.0 / 255.0)
    return image

# -----------------------------------------------------------------------------------
def read_image_sets(images_dir, width, height, hard_load, verbose = False):
    images = []
    labels = []
    categories = sorted(os.listdir(images_dir))
    L = len(categories)

    for i in range(L):
        path  = os.path.join(images_dir, categories[i], '*g') # *g for png, jpg, jpeg
        subdir= glob.glob(path) #[:2000]
        if verbose: print("Folder",categories[i],"contains",len(subdir),"images")
        for filename in subdir:
            if hard_load == True:    # load all data into memory
                image = load_image(filename, width, height)
                images.append(image)
                label = np.zeros(L)
                label[i] = 1.0
                labels.append(label)
            else:                   # data will be loaded during training
                images.append(filename)
                labels.append(i)

    images, labels = sklearn.utils.shuffle(images, labels)

    return images, labels

# -----------------------------------------------------------------------------------
class Data(object):
    def __init__(self, images, labels, width, height, L, hard_load):
        self.L        = L
        self.epoch    = 0
        self.images   = images
        self.labels   = labels
        self.size     = len(self.images)
        self.width    = width
        self.height   = height
        self.hard_load= hard_load

    # Return the next `batch_size` examples from this data set.
    def next_batch(self, batch_size):
        start = self.epoch
        self.epoch += batch_size

        if self.epoch > self.size:
          start = 0
          self.epoch = batch_size
          assert batch_size <= self.size
        end = self.epoch

        if self.hard_load == True:
            return self.images[start:end], self.labels[start:end]
        else:
            images = []
            labels = []
            for i in range(start,end):
                image = load_image(self.images[i], self.width, self.height)
                images.append(image)
                label = np.zeros(self.L)
                label[self.labels[i]] = 1.0
                labels.append(label)
            return images, labels

# -----------------------------------------------------------------------------------
class DataSet(Data):
    def __init__(self, images_dir, width, height, hard_load, split_ratio = 0.0, verbose = False):
        outs = read_image_sets(images_dir, width, height, hard_load, verbose)
        L    = len(os.listdir(images_dir))
        if split_ratio == 0.0:
            Data.__init__(self, outs[0], outs[1], width, height, L, hard_load)
        else:
            ratio      = int(split_ratio * len(outs[0]))
            self.train = Data(outs[0][ratio:], outs[1][ratio:], width, height, L, hard_load)
            self.valid = Data(outs[0][:ratio], outs[1][:ratio], width, height, L, hard_load)
