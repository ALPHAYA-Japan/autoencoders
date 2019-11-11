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
def load_splitted_image(filename, width, height):
    image   = cv2.imread(filename)
    image   = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    w       = image.shape[1]    # get actual width
    image_L = image[:,:w >> 1,:]
    image_R = image[:,w >> 1:,:]
    image_L = cv2.resize(image_L, (height, width),0,0, cv2.INTER_LINEAR)
    image_R = cv2.resize(image_R, (height, width),0,0, cv2.INTER_LINEAR)
    image_L = np.multiply(image_L.astype(np.float32), 1.0 / 255.0)
    image_R = np.multiply(image_R.astype(np.float32), 1.0 / 255.0)
    return image_L, image_R

# -----------------------------------------------------------------------------------
def read_image_sets(images_dir, width, height, hard_load, verbose = False):
    images = []
    labels = []
    categories = sorted(os.listdir(images_dir))
    L = len(categories)

    for i in range(L):
        path  = os.path.join(images_dir, categories[i], '*g') # *g for png, jpg, jpeg
        subdir= glob.glob(path)
        if verbose: print("Folder",categories[i],"contains",len(subdir),"images")
        for filename in subdir:
            if hard_load == True:    # load all data into memory
                image = load_splitted_image(filename, width, height)
                images.append(image[0])
                labels.append(image[1])
            else:                   # data will be loaded during training
                images.append(filename)
                labels.append(filename)

    images, labels = sklearn.utils.shuffle(images, labels)

    return images, labels

# -----------------------------------------------------------------------------------
class Data(object):
    def __init__(self, images, labels, width, height, hard_load):
        self.epoch    = 0
        self.images   = images
        self.labels   = labels
        self.size     = len(self.images)
        self.width    = width
        self.height   = height
        self.hard_load= hard_load

    """Return the next `batch_size` examples from this data set."""
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
                image = load_splitted_image(self.images[i], self.width, self.height)
                images.append(image[0])
                labels.append(image[1])
            return images, labels

# -----------------------------------------------------------------------------------
class DataSet(Data):
    def __init__(self, images_dir, width, height, hard_load, split_ratio = 0.0, verbose = False):
        outs = read_image_sets(images_dir, width, height, hard_load, verbose)
        if split_ratio == 0.0:
            Data.__init__(self, outs[0], outs[1], width, height, hard_load)
        else:
            ratio      = int(split_ratio * len(outs[0]))
            self.train = Data(outs[0][ratio:], outs[1][ratio:], width, height, hard_load)
            self.valid = Data(outs[0][:ratio], outs[1][:ratio], width, height, hard_load)
