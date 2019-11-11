'''
    ------------------------------------
    Author : SAHLI Mohammed
    Date   : 2019-11-09
    Company: Alphaya (www.alphaya.com)
    Email  : nihon.sahli@gmail.com
    ------------------------------------
'''

import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages

# flatten the input..................................................................
def flatten(input,verbose=False):
    # We know that the shape of the input will be
    # [batch_size image_size image_size channels]
    # But let's get it from the previous input.
    shape = input.get_shape()

    # Number of features will be img_height * img_width * channels.
    # But we shall calculate it in place of hard-coding it.
    num_features = shape[1:4].num_elements()

    # Now, we Flatten the input so we shall have to reshape to num_features
    h = tf.reshape(input, [-1, num_features])
    if verbose: print(h.shape)
    return h


#................................................................................
# Get Weights
#................................................................................
def get_weights(shape, stddev = 0.02, name = 'weight', initializer = 'truncated'):
    input_dim = int(shape[0])
    output_dim= int(shape[1])
    if initializer == 'truncated':
        initializer = tf.truncated_normal_initializer(stddev = stddev)
        return tf.get_variable(name, shape, initializer = initializer)
    elif initializer == "xavier":
        val = 1 * np.sqrt(6.0 / (output_dim + input_dim))                         # You can use 4 for sigmoid or 1 for tanh activation
        initial = tf.random_uniform([input_dim, output_dim],                    #
                                     minval =-val,                              # Minimum Value
                                     maxval = val,                              # Maximum Value
                                     dtype  = tf.float32)
        return tf.Variable(initial, name = name)
    elif initializer == "uniform":
        val = int(np.sqrt(6.0 / (output_dim + input_dim)))
        initial = tf.random_uniform([input_dim, output_dim],
                                     minval =-val,                              # Minimum Value
                                     maxval = val,                              # Maximum Value
                                     dtype  = tf.float32)
        return tf.Variable(initial, name = name)
    elif initializer == "gaussian":
        val = 2.0 / (output_dim * input_dim)
        initial = tf.random_normal([input_dim, output_dim],
                                    mean    = 0,                                # Mean Value
                                    stddev  = val,                              # Standard Deviation
                                    dtype   = tf.float32)
        return tf.Variable(initial, name = name)
    elif initializer == "glorot":
        shape   = [input_dim, output_dim]
        initial = tf.random_normal(shape = shape,
                                   stddev= 1.0 / tf.sqrt(shape[0] / 2.0))
        return tf.Variable(initial, name = name)
    elif initializer == "normal":
        initializer = tf.random_normal_initializer(stddev=stddev)
        return tf.get_variable(name, shape, initializer = initializer)
    else:
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05),name=name)

#................................................................................
# Get Biases
#................................................................................
def get_biases(shape, constant = 0.05, name = 'bias', initializer = "constant"):
    if initializer == "constant":
        # initializer = tf.constant(constant, shape = shape)
        # return tf.Variable(initializer, name = name)
        initializer = tf.constant_initializer(constant)
        return tf.get_variable(name, shape, initializer=initializer)
    elif initializer == "normal":
        return tf.Variable(tf.random_normal(shape), name = name)
    else:
        return tf.Variable(tf.zeros(shape), name = name)

#................................................................................
# Batch Normalization Layer
# This is the same as tf.contrib.layers.batch_norm
#................................................................................
def batch_norm(x, scope, is_training, epsilon=1e-5, decay = 0.9):
    with tf.variable_scope(scope, reuse=False):
        shape  = x.get_shape().as_list()
        gamma  = tf.get_variable("gamma" , shape[-1])
        beta   = tf.get_variable("beta"  , shape[-1])
        mv_avg = tf.get_variable("mv_avg", shape[-1], trainable=False)
        mv_var = tf.get_variable("mv_var", shape[-1], trainable=False)
        control_inputs = []
        if is_training:
            axes = [x for x in range(len(shape)-1)]
            avg, var      = tf.nn.moments(x, axes)
            update_mv_avg = moving_averages.assign_moving_average(mv_avg,avg,decay)
            update_mv_var = moving_averages.assign_moving_average(mv_var,var,decay)
            control_inputs= [update_mv_avg, update_mv_var]
        else:
            avg = mv_avg
            var = mv_var
        with tf.control_dependencies(control_inputs):
            output = tf.nn.batch_normalization(x, avg, var, offset=beta, scale=gamma, variance_epsilon=epsilon)
    return output

#................................................................................
# Convolutional 2D Layer
#................................................................................
def conv2d(x, output_dim, kernel_size=[4,4], strides=[1,2,2,1], name="conv2d", w_initializer = 'truncated', b_initializer = 'constant'):
    with tf.variable_scope(name):
        shape = [kernel_size[0], kernel_size[1], int(x.get_shape()[-1]), output_dim]
        w = get_weights(shape      , initializer = w_initializer)
        b = get_biases([output_dim], initializer = b_initializer)
        r = tf.nn.conv2d(x, w, strides = strides, padding='SAME')
        r = tf.nn.bias_add(r, b)
        return r

#................................................................................
# Deconvolutional 2D Layer
#................................................................................
def deconv2d(x, output_shape, kernel_size=[4,4], strides=[1,2,2,1], name="deconv2d", w_initializer = 'truncated', b_initializer = 'constant'):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        shape = [kernel_size[0], kernel_size[1], output_shape[-1], x.get_shape()[-1]]
        w = get_weights(shape           , initializer = w_initializer)
        b = get_biases([output_shape[-1]], initializer = b_initializer)
        r = tf.nn.conv2d_transpose(x, w, output_shape = output_shape, strides = strides)
        r = tf.nn.bias_add(r, b)
        # r = tf.reshape(tf.nn.bias_add(r, b), r.get_shape())
        return r

#................................................................................
# Linear Layer
#................................................................................
def linear(x, output_size, scope=None, stddev=0.02, bias_start=0.0, w_initializer = 'normal', b_initializer = 'constant'):
    shape = x.get_shape().as_list()
    with tf.variable_scope(scope or "Linear"):
        kernel_shape = [int(shape[1]), output_size]
        w = get_weights(kernel_shape, initializer = w_initializer)
        b = get_biases([output_size], initializer = b_initializer)
        r = tf.matmul(x, w) + b
        return r
