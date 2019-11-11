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
import sys
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
sys.path.append('..')
import utils.loader as loader
import utils.utils  as utils
import utils.layers as layers

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Adding Seed so that random initialization is consistent
from tensorflow import set_random_seed
np.random.seed(1)
set_random_seed(2)

class CONV_AE:
    #................................................................................
    # Constructor
    #................................................................................
    def __init__(self, model_path, data_path = None, is_training = False,
                 batch_size = 64, image_size = 28, latent_dim = 200,
                 hard_load = False, verbose = False, pretrained = False):
        self.pretrained   = pretrained
        self.image_size   = image_size  # height and weight of images
        self.channels     = 3           # dealing with rgb images, hence 3 channels
        self.batch_size   = batch_size
        self.latent_dim   = latent_dim
        self.linear_dim   = 1024
        self.conv_dim     = 128
        self.lr_rate      = 1e-4        # or 1e-3
        self.beta1        = 0.5
        self.w_initializer= "xavier"    # or "uniform", "gaussian", "truncated"
        self.b_initializer= "constant"  # or "normal"

        self.verbose      = verbose
        self.model_path   = model_path
        self.train_path   = data_path
        self.session      = tf.Session()

        # CONV_AE Parameters
        self.model_name   = "CONV"
        self.validation_size= 0.20      # 20% of the data will be used for validation

        if is_training == True:
            self.train_initialization(hard_load     = hard_load,
                                      en_iterations = 1,
                                      de_iterations = 1)
        else:
            self.predict_initialization()

    #................................................................................
    # Deconstructor
    #................................................................................
    def __del__(self):
        pass

    #................................................................................
    # Training Initialization
    #................................................................................
    def train_initialization(self, hard_load = True, en_iterations = 1, de_iterations = 1):
        # initialize training dataset
        self.data = loader.DataSet(images_dir = self.train_path,
                                   width      = self.image_size,
                                   height     = self.image_size,
                                   split_ratio= self.validation_size,
                                   hard_load  = hard_load            )

        self.en_iterations = max(1, en_iterations)
        self.de_iterations = max(1, de_iterations)

        # get number of batches for a single epoch
        self.num_batches = self.data.train.size // self.batch_size

        if self.verbose:
            print("Train Data size=", self.data.train.size)
            print("Test  Data size=", self.data.valid.size)
            print("Batch size     =", self.batch_size     )
            print("Learning rate  =", self.lr_rate        )

        self.input_shape = [None, self.image_size, self.image_size, self.channels]
        self.en_input    = tf.placeholder(tf.float32, shape = self.input_shape, name = 'en_input')

        # create the network's model and optimizer
        self.create_network()
        self.create_optimizer()

        # initialize of all global variables
        global_variables = tf.global_variables_initializer()
        self.session.run(global_variables)
        self.saver = tf.train.Saver()

        if self.pretrained == True:
            if self.verbose == True: print("Loading pretrained model...",end='')
            meta_graph = self.model_path + self.model_name + '.meta'
            checkpoint = tf.train.latest_checkpoint(self.model_path)  #
            self.saver.restore(self.session, checkpoint)              # Load the weights
            if self.verbose == True: print("done")

    #................................................................................
    # Prediction Initialization
    #................................................................................
    def predict_initialization(self):
        meta_graph = self.model_path + self.model_name + '.meta'
        self.saver = tf.train.import_meta_graph(meta_graph)       # Recreate the network graph
        checkpoint = tf.train.latest_checkpoint(self.model_path)  #
        self.saver.restore(self.session, checkpoint)              # Load the weights
        self.graph = tf.get_default_graph()

    #................................................................................
    #
    #................................................................................
    def encoder(self, x, is_training=True, reuse=False):
        with tf.variable_scope("encoder", reuse=reuse):
            if self.verbose: print(x.shape)

            net = tf.keras.layers.Conv2D(filters    = 32,
                                        kernel_size = [5,5],
                                        strides     = 2,
                                        padding     = 'same',
                                        kernel_initializer='glorot_uniform')(x)
            if self.verbose: print(net.shape)

            # net = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(net)
            # if self.verbose: print(net.shape)

            net = tf.keras.layers.Conv2D(filters    = 16,
                                        kernel_size = [5,5],
                                        strides     = 2,
                                        padding     = 'same',
                                        kernel_initializer='glorot_uniform')(net)
            if self.verbose: print(net.shape)

            # net = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(net)
            # if self.verbose: print(net.shape)

            # net = tf.keras.layers.Conv2D(filters    = 8,
            #                             kernel_size = [5,5],
            #                             strides     = 4,
            #                             padding     = 'same',
            #                             kernel_initializer='glorot_uniform')(net)
            # if self.verbose: print(net.shape)
            self.main_shape = [-1,int(net.shape[1]),int(net.shape[2]),int(net.shape[3])] # will be used by the decoder

            # net = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(net)
            # if self.verbose: print(net.shape)

            net = layers.flatten(net)
            if self.verbose: print(net.shape)

            net = layers.linear(net, self.latent_dim << 2, scope = 'en_fc1',
                                w_initializer = self.w_initializer,
                                b_initializer = self.b_initializer)
            net = tf.nn.relu(net)
            if self.verbose: print(net.shape)

            net = layers.linear(net, self.latent_dim << 1, scope = 'en_fc2',
                                w_initializer = self.w_initializer,
                                b_initializer = self.b_initializer)
            net = tf.nn.relu(net)
            if self.verbose: print(net.shape)

            net = layers.linear(net, self.latent_dim, scope = 'en_fc3',
                                w_initializer = self.w_initializer,
                                b_initializer = self.b_initializer)
            net = tf.nn.relu(net, name = 'main_out')
            if self.verbose: print(net.shape)

            return net

    #................................................................................
    # Discriminator Architecture
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    #................................................................................
    def encoder_v2(self, x, is_training=True, reuse=False):
        with tf.variable_scope("encoder_v2", reuse = reuse):
            if self.verbose: print(x.shape)

            # Layer 1
            net = layers.conv2d(x, self.conv_dim >> 1, name = 'en_conv1')
            net = tf.nn.leaky_relu(net)
            if self.verbose: print(net.shape)

            # Layer 2
            net = layers.conv2d(net, self.conv_dim, name = 'en_conv2')
            net = layers.batch_norm(net, is_training = is_training, scope = 'en_bn2')
            net = tf.nn.leaky_relu(net)
            if self.verbose: print(net.shape)

            # Layer 3
            net = layers.flatten(net)
            if self.verbose: print(net.shape)

            # Layer 4
            net = layers.linear(net, self.linear_dim, scope = 'en_fc3')
            net = layers.batch_norm(net, is_training = is_training, scope = 'en_bn3')
            net = tf.nn.leaky_relu(net)
            if self.verbose: print(net.shape)

            # Layer 5
            out_logit = layers.linear(net, self.latent_dim, scope = 'en_fc4')
            out = tf.nn.sigmoid(out_logit, name = "main_out")
            if self.verbose: print(out.shape)

            return out

    #................................................................................
    #
    #................................................................................
    def decoder(self, x, is_training=True, reuse=False):
        with tf.variable_scope("decoder", reuse=reuse):
            net = layers.linear(x, self.latent_dim << 1, scope = 'de_fc1',
                                w_initializer = self.w_initializer,
                                b_initializer = self.b_initializer)
            net = tf.nn.relu(net)
            if self.verbose: print(net.shape)

            net = layers.linear(net, self.latent_dim << 2, scope = 'de_fc2',
                                w_initializer = self.w_initializer,
                                b_initializer = self.b_initializer)
            net = tf.nn.relu(net)
            if self.verbose: print(net.shape)

            out_dim = self.main_shape[1] * self.main_shape[2] * self.main_shape[3]
            net = layers.linear(net, out_dim, scope = 'de_fc3',
                                w_initializer = self.w_initializer,
                                b_initializer = self.b_initializer)
            net = tf.nn.relu(net)
            if self.verbose: print(net.shape)

            net = tf.reshape(net,self.main_shape)
            if self.verbose: print(net.shape)

            # net = tf.keras.layers.Conv2DTranspose(filters = 16,
            #                                      kernel_size = [5,5],
            #                                      strides=4,
            #                                      padding='same',
            #                                      kernel_initializer='glorot_uniform')(net)
            # if self.verbose: print(net.shape)

            # net = tf.keras.layers.UpSampling2D((2, 2))(net)
            # if self.verbose: print(net.shape)

            net = tf.keras.layers.Conv2DTranspose(filters = 32,
                                                 kernel_size = [5,5],
                                                 strides=2,
                                                 padding='same',
                                                 kernel_initializer='glorot_uniform')(net)
            if self.verbose: print(net.shape)

            # net = tf.keras.layers.UpSampling2D((2, 2))(net)
            # if self.verbose: print(net.shape)

            net = tf.keras.layers.Conv2DTranspose(filters = 3,
                                                 kernel_size = [5,5],
                                                 strides=2,
                                                 padding='same',
                                                 kernel_initializer='glorot_uniform')(net)
            if self.verbose: print(net.shape)

            ## net = tf.keras.layers.UpSampling2D((2, 2))(net)
            # if self.verbose: print(net.shape)

            net = tf.sigmoid(net, name = 'main_out')

            return net

    #................................................................................
    # Decoder Architecture is the same as Generator Architecture
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    #................................................................................
    def decoder_v2(self, x, is_training=True, reuse=False):
        with tf.variable_scope("decoder", reuse=reuse):
            if self.verbose: print(x.shape)

            # Layer 1
            net = layers.linear(x, self.linear_dim, scope = 'de_fc1')
            net = layers.batch_norm(net, is_training = is_training, scope = 'de_bn1')
            net = tf.nn.relu(net)
            if self.verbose: print(net.shape)

            # Layer 2
            shape = self.conv_dim * ((self.image_size >> 2) ** 2)
            net = layers.linear(net, shape, scope = 'de_fc2')
            net = layers.batch_norm(net, is_training = is_training, scope = 'de_bn2')
            net = tf.nn.relu(net)
            if self.verbose: print(net.shape)

            # Layer 3
            shape = [self.batch_size, self.image_size >> 2, self.image_size >> 2, self.conv_dim]
            net = tf.reshape(net, shape)
            if self.verbose: print(net.shape)

            # Layer 4
            shape = [self.batch_size, self.image_size >> 1, self.image_size >> 1, self.conv_dim >> 1]
            net = layers.deconv2d(net, shape, name = 'de_dc3')
            net = layers.batch_norm(net, is_training = is_training, scope = 'de_bn3')
            net = tf.nn.relu(net)
            if self.verbose: print(net.shape)

            # Layer 5
            shape = [self.batch_size, self.image_size, self.image_size, self.channels]
            net = layers.deconv2d(net, shape, name = 'de_dc4')
            out = tf.nn.sigmoid(net, name = "main_out")
            if self.verbose: print(out.shape)

            return out

    #................................................................................
    #
    #................................................................................
    def create_network(self):
        self.encoded = self.encoder(self.en_input, is_training = True, reuse = False)
        self.decoded = self.decoder(self.encoded , is_training = True, reuse = False)

    #................................................................................
    #
    #................................................................................
    def create_optimizer(self):
        # Loss
        self.de_loss = tf.reduce_mean(tf.pow(self.en_input - self.decoded, 2))

        # Optimizer
        ae_optimizer = tf.train.AdamOptimizer(self.lr_rate, beta1 = self.beta1)

        # Training Variables
        t_vars = tf.trainable_variables()

        # Create training operations
        self.de_opt  = ae_optimizer.minimize(self.de_loss, var_list = t_vars)

    #................................................................................
    #
    #................................................................................
    def train(self, max_epoches, show_images = False):
        prev_t_loss= sys.float_info.max # set it to the maximum
        prev_v_loss= sys.float_info.max # set it to the maximum
        f, a = plt.subplots(4, 10, figsize=(10, 4))

        for epoch in range(max_epoches):
            for i in range(self.num_batches):
                x_batch, _ = self.data.train.next_batch(self.batch_size)
                _, t_loss = self.session.run([self.de_opt, self.de_loss],
                                              feed_dict = {self.en_input: x_batch})

                if i % 100 == 0:
                    x_v_batch, _ = self.data.valid.next_batch(self.batch_size)
                    v_loss = self.session.run(self.de_loss,
                                              feed_dict = {self.en_input: x_v_batch})

                    msg = "Epoch {}-{}/{}\tt_loss: {:.5f}\tv_loss: {:.5f}"
                    print(msg.format(epoch+1, i+1, self.num_batches, t_loss, v_loss))
                    if math.isnan(t_loss) == False and math.isnan(v_loss) == False:
                        if t_loss <= prev_t_loss and v_loss <= prev_v_loss:
                            prev_t_loss = t_loss
                            prev_v_loss = v_loss
                            self.saver.save(self.session, self.model_path + self.model_name)
                            print("recent model was saved to",self.model_path + self.model_name)
                    else:
                        sys.exit()

                    if show_images == True:
                        g = self.session.run(self.decoded,
                                             feed_dict = {self.en_input: x_v_batch})
                        for j in range(10):
                            a[0][j].imshow(x_v_batch[j])
                            a[1][j].imshow(g[j])
                            a[2][j].imshow(x_v_batch[10+j])
                            a[3][j].imshow(g[10+j])

                        f.suptitle("Epoch "+str(epoch)+", Step "+str(i), fontsize=9)
                        f.show()
                        plt.draw()
                        plt.pause(0.001)

            preds = self.session.run(self.decoded,
                                     feed_dict = {self.en_input: x_v_batch})
            grid = self.construct_image_grid(x_v_batch, preds, 20, 480, 240)
            cv2.imwrite("images/"+self.model_name+"/grid_" + str(epoch+1) + ".png", grid)

        plt.close()

    #................................................................................
    #
    #................................................................................
    def construct_image_grid(self, batch, preds, samples, grid_width, grid_height):
        for i in range(samples):
            batch[i] = utils.add_border(batch[i],color = [1.0,0.0,0.0])
        N    = samples >> 1
        grid = [np.concatenate(tuple(batch[ :N]     ), axis = 1),
                np.concatenate(tuple(preds[ :N]     ), axis = 1),
                np.concatenate(tuple(batch[N:N << 1]), axis = 1),
                np.concatenate(tuple(preds[N:N << 1]), axis = 1)]
        grid = np.concatenate(tuple(grid), axis = 0)
        grid = cv2.resize(grid, (grid_width, grid_height),
                          interpolation = cv2.INTER_AREA)
        grid  = (grid * 255.0).astype(np.uint8)
        return grid

    #................................................................................
    #
    #................................................................................
    def generate(self, source, destination = None, samples = 20, grid_width=480, grid_height=240):
        # Load the input and output of the graph
        input  = self.graph.get_tensor_by_name("en_input:0"        )
        output = self.graph.get_tensor_by_name("decoder/main_out:0")

        # For generating a single image for a source image
        if os.path.isfile(source):
            image = loader.load_image(source, self.image_size, self.image_size)
            batch = np.asarray([image for _ in range(self.batch_size)])
            preds = self.session.run(output, feed_dict = {input: batch})
            batch[0] = utils.add_border(batch[0],color = [1.0,0.0,0.0])
            grid  = np.concatenate((batch[0], preds[0]), axis=1)
            grid  = (grid * 255.0).astype(np.uint8)
        # For generating a grid of images from a directory of images
        elif os.path.isdir(source):
            data = loader.DataSet(images_dir = source,
                                  hard_load  = False,
                                  width      = self.image_size,
                                  height     = self.image_size)
            batch,_= data.next_batch(self.batch_size)

            preds  = self.session.run(output, feed_dict={input: batch})
            for i in range(samples):
                batch[i] = utils.add_border(batch[i],color = [1.0,0.0,0.0])
            grid = self.construct_image_grid(batch,preds,samples,grid_width,grid_height)
        else:
            print(source,"must be an image pathname or a directory of images")
            sys.exit()

        if destination:
            cv2.imwrite(destination, grid)
        else:
            cv2.imshow("images", grid)
            cv2.waitKey()
