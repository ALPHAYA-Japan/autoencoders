'''
    ------------------------------------
    Author : SAHLI Mohammed
    Date   : 2019-11-09
    Company: Alphaya (www.alphaya.com)
    Email  : nihon.sahli@gmail.com
    Credits: https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
    ------------------------------------
'''

import numpy as np

# 'gauss'     Gaussian-distributed additive noise.
# 'poisson'   Poisson-distributed noise generated from the data.
# 's_and_p'   Replaces random pixels with 0 or 1.
# 'speckle'   Multiplicative noise using out = image + n*image,where
#             n is uniform noise with specified mean & variance.
def noisy(noise_type,image):
    if noise_type == "gaussian":
        row,col,ch= image.shape
        mean  = 0
        var   = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        min_  = noisy.min()
        max_  = noisy.max()
        noisy = (noisy - min_) / (max_ - min_)
        return noisy
    elif noise_type == "s_and_p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out    = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[tuple(coords)] = 1
        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[tuple(coords)] = 0
        min_  = out.min()
        max_  = out.max()
        out = (out - min_) / (max_ - min_)
        return out
    elif noise_type == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        min_  = noisy.min()
        max_  = noisy.max()
        noisy = (noisy - min_) / (max_ - min_)
        return noisy
    elif noise_type =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)
        noisy = image + image * gauss
        min_  = noisy.min()
        max_  = noisy.max()
        noisy = (noisy - min_) / (max_ - min_)
        return noisy
    return image

# add border color to an image
def add_border(image, color):
    image[0,:,:]  = color
    image[:,0,:]  = color
    image[-1,:,:] = color
    image[:,-1,:] = color
    return image
