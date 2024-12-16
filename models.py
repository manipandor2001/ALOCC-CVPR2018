from __future__ import division
#from tensorflow.examples.tutorials.mnist import input_data
import re
from ops import *
from utils import *
from kh_tools import *
import logging
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
import logging
from glob import glob
import random

class ALOCC_Model(object):
    def __init__(self, 
                 input_height=45, input_width=45, output_height=64, output_width=64,
                 batch_size=128, sample_num=128, attention_label=1, is_training=True,
                 z_dim=100, gf_dim=16, df_dim=16, gfc_dim=512, dfc_dim=512, c_dim=3,
                 dataset_name=None, dataset_address=None, input_fname_pattern=None,
                 checkpoint_dir=None, log_dir=None, sample_dir=None, r_alpha=0.2,
                 kb_work_on_patch=True, nd_input_frame_size=(240, 360), nd_patch_size=(10, 10), n_stride=1,
                 n_fetch_data=10, n_per_itr_print_results=500):

        self.n_per_itr_print_results = n_per_itr_print_results
        self.nd_input_frame_size = nd_input_frame_size
        self.b_work_on_patch = kb_work_on_patch
        self.sample_dir = sample_dir
 
        self.is_training = is_training
        self.r_alpha = r_alpha
        self.batch_size = batch_size
        self.sample_num = sample_num
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.z_dim = z_dim
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim
        self.attention_label = attention_label
        self.dataset_name = dataset_name
        self.dataset_address = dataset_address
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

        # Initialize batch normalization layers using tf.keras.layers.BatchNormalization
        self.g_bn0 = tf.keras.layers.BatchNormalization()
        self.g_bn1 = tf.keras.layers.BatchNormalization()
        self.g_bn2 = tf.keras.layers.BatchNormalization()
        self.g_bn3 = tf.keras.layers.BatchNormalization()
        self.g_bn4 = tf.keras.layers.BatchNormalization()
        self.g_bn5 = tf.keras.layers.BatchNormalization()
        self.g_bn6 = tf.keras.layers.BatchNormalization()

        self.d_bn1 = tf.keras.layers.BatchNormalization()
        self.d_bn2 = tf.keras.layers.BatchNormalization()
        self.d_bn3 = tf.keras.layers.BatchNormalization()
        self.d_bn4 = tf.keras.layers.BatchNormalization()

        if self.dataset_name == 'mnist':
            # For MNIST dataset
            from tensorflow.examples.tutorials.mnist import input_data
            mnist = input_data.read_data_sets(self.dataset_address)
            specific_idx = np.where(mnist.train.labels == self.attention_label)[0]
            self.data = mnist.train.images[specific_idx].reshape(-1, 28, 28, 1)
            self.c_dim = 1
        elif self.dataset_name == 'UCSD':
            # For UCSD dataset
            self.nStride = n_stride
            self.patch_size = nd_patch_size
            self.patch_step = (n_stride, n_stride)
            lst_image_paths = []

            

            for s_image_dir_path in glob(os.path.join(self.dataset_address, self.input_fname_pattern)):


              for sImageDirFiles in glob(os.path.join(s_image_dir_path + '/*')):
                  lst_image_paths.append(sImageDirFiles)
            self.dataAddress = lst_image_paths

            print(len(lst_image_paths)) #debugging 
            print(f'n_fetch_data : {n_fetch_data:.2f}')


            lst_forced_fetch_data = [self.dataAddress[x] for x in random.sample(range(0, len(lst_image_paths)), n_fetch_data)]
            self.data = lst_forced_fetch_data
            self.c_dim = 1
        else:
            assert('Error in loading dataset')



        self.generator = generator()
        self.discriminator = discriminator()

       
        self.model_dir = model_dir()
      
        

    

        self.g_optimizer = RMSprop(learning_rate=config.learning_rate)
        self.d_optimizer = RMSprop(learning_rate=config.learning_rate)

        self.checkpoint = tf.train.Checkpoint(generator=self.generator, 
                                              discriminator=self.discriminator, 
                                              g_optimizer=self.g_optimizer, 
                                              d_optimizer=self.d_optimizer)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, 
                                                             directory=config.checkpoint_dir, 
                                                             max_to_keep=5)

        self.grayscale = (self.c_dim == 1)
        self.build_model()
    def train_step(self, batch_images, batch_noise_images):
        batch_z = tf.random.uniform([batch_images.shape[0], self.z_dim], -1, 1)

        # Train discriminator
        with tf.GradientTape() as d_tape:
            fake_images = self.generator(batch_noise_images, training=True)
            real_logits = self.discriminator(batch_images, training=True)
            fake_logits = self.discriminator(fake_images, training=True)

            d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_logits), logits=real_logits))
            d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_logits), logits=fake_logits))
            d_loss = d_loss_real + d_loss_fake

        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))

        # Train generator
        with tf.GradientTape() as g_tape:
            fake_images = self.generator(batch_noise_images, training=True)
            fake_logits = self.discriminator(fake_images, training=True)
            g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_logits), logits=fake_logits))

        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))

        return d_loss, g_loss

    def train(self):
        # Initialize logging directories
        os.makedirs(self.sample_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        train_dataset = tf.data.Dataset.from_tensor_slices(self.data)
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(self.config.batch_size)

        # Resume training from a checkpoint if exists
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print("Checkpoint restored.")

        for epoch in range(self.config.epochs):
            print(f"Epoch ({epoch + 1}/{self.config.epochs})-------------------------------")

            for step, batch_data in enumerate(train_dataset):
                # Assuming `batch_noise` is prepared the same way for training.
                batch_noise = self.prepare_noisy_batch(batch_data)
                d_loss, g_loss = self.train_step(batch_data, batch_noise)

                print(f"Step {step}, D Loss: {d_loss.numpy():.4f}, G Loss: {g_loss.numpy():.4f}")

                if step % 100 == 0:  # Save sample images or model checkpoints as necessary
                    self.generate_samples(epoch, step)

            # Save checkpoint at the end of each epoch
            self.checkpoint_manager.save()
            print(f"Checkpoint saved for epoch {epoch + 1}.")

    def prepare_noisy_batch(self, batch_data):
        """Apply noise to data for training."""
        noise = tf.random.normal(shape=batch_data.shape, mean=0.0, stddev=0.1)
        return batch_data + noise

    def generate_samples(self, epoch, step):
        """Generate and save sample images from the generator."""
        sample_z = tf.random.uniform([self.config.sample_num, self.z_dim], -1, 1)
        generated_samples = self.generator(sample_z, training=False)

        # Save samples as images
        for idx, sample in enumerate(generated_samples):
            save_path = os.path.join(self.sample_dir, f"sample_epoch_{epoch}_step_{step}_idx_{idx}.png")
            tf.keras.preprocessing.image.save_img(save_path, sample.numpy())
    
    def generator(z, output_height, output_width, df_dim, gf_dim, c_dim, batch_size):
        s_h, s_w = output_height, output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # Batch normalization layers
        g_bn2 = tf.keras.layers.BatchNormalization()
        g_bn3 = tf.keras.layers.BatchNormalization()
        g_bn4 = tf.keras.layers.BatchNormalization()
        g_bn5 = tf.keras.layers.BatchNormalization()
        g_bn6 = tf.keras.layers.BatchNormalization()

        # Encoder stages
        hae0 = tf.keras.layers.ReLU()(g_bn4(tf.keras.layers.Conv2D(df_dim * 2, kernel_size=3, strides=2, padding='same')(z)))
        hae1 = tf.keras.layers.ReLU()(g_bn5(tf.keras.layers.Conv2D(df_dim * 4, kernel_size=3, strides=2, padding='same')(hae0)))
        hae2 = tf.keras.layers.ReLU()(g_bn6(tf.keras.layers.Conv2D(df_dim * 8, kernel_size=3, strides=2, padding='same')(hae1)))

        # Decoder stages (Deconvolution / Upsample)
        h2 = tf.keras.layers.ReLU()(g_bn2(tf.keras.layers.Conv2DTranspose(gf_dim * 2, kernel_size=3, strides=2, padding='same')(hae2)))
        h3 = tf.keras.layers.ReLU()(g_bn3(tf.keras.layers.Conv2DTranspose(gf_dim, kernel_size=3, strides=2, padding='same')(h2)))
        h4 = tf.keras.layers.Conv2DTranspose(c_dim, kernel_size=3, strides=2, padding='same')(h3)

        return tf.nn.tanh(h4)

        def discriminator(self, image,reuse=False):
          with tf.variable_scope("discriminator") as scope:
            if reuse:
              scope.reuse_variables()


          h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
          h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
          h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
          h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
          h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
          h5 = tf.nn.sigmoid(h4,name='d_output')
          return h5, h4





    def build_model(self):
        image_dims = [self.input_height, self.input_width, self.c_dim]

        # Use TensorFlow 2.x's model building using layers
        self.inputs = tf.keras.Input(shape=(self.batch_size,) + tuple(image_dims), dtype=tf.float32)
        self.sample_inputs = tf.keras.Input(shape=(self.sample_num,) + tuple(image_dims), dtype=tf.float32)

        self.z = tf.keras.Input(shape=(self.batch_size,) + tuple(image_dims), dtype=tf.float32)

        # Define the model layers here (you'll replace the original `generator` and `discriminator` methods)
        self.G = self.generator(self.z)
        self.D, self.D_logits = self.discriminator(self.inputs)

        self.sampler = self.sampler(self.z)
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        # Calculate losses
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))

        # Refinement loss
        self.g_r_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.G, labels=self.z))
        self.g_loss = self.g_loss + self.g_r_loss * self.r_alpha
        self.d_loss = self.d_loss_real + self.d_loss_fake

        # Using TensorFlow 2.x summaries
        with tf.summary.create_file_writer(self.log_dir).as_default():
            tf.summary.scalar("d_loss_real", self.d_loss_real)
            tf.summary.scalar("d_loss_fake", self.d_loss_fake)
            tf.summary.scalar("g_loss", self.g_loss)
            tf.summary.scalar("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

