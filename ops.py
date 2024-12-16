"""
Some code used from from https://github.com/Newmu/dcgan_code
"""

import math
import numpy as np 
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *

import tensorflow as tf


# TensorFlow 2.x summarization replacements
#image_summary = tf.summary.image
#scalar_summary = tf.summary.scalar
#histogram_summary = tf.summary.histogram
#merge_summary = tf.summary.merge
#SummaryWriter = tf.summary.create_file_writer


# Concatenation remains the same in TensorFlow 2.x
def concat(tensors, axis, *args, **kwargs):
    return tf.concat(tensors, axis, *args, **kwargs)


# Batch normalization using TensorFlow 2.x's tf.keras.layers.BatchNormalization
class batch_norm:
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        self.batch_norm_layer = tf.keras.layers.BatchNormalization(
            epsilon=epsilon, momentum=momentum, name=name
        )

    def __call__(self, x, train=True):
        return self.batch_norm_layer(x, training=train)


# Concatenate conditioning vector on the feature map axis
def conv_cond_concat(x, y):
    x_shapes = tf.shape(x)
    y_shapes = tf.shape(y)
    y_tiled = tf.tile(y, [x_shapes[0], x_shapes[1], x_shapes[2], 1])
    return tf.concat([x, y_tiled], axis=-1)


# 2D Convolution
def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
    initializer = tf.random_normal_initializer(stddev=stddev)
    conv_layer = tf.keras.layers.Conv2D(
        filters=output_dim,
        kernel_size=(k_h, k_w),
        strides=(d_h, d_w),
        padding="same",
        kernel_initializer=initializer,
        bias_initializer=tf.zeros_initializer(),
        name=name,
    )
    return conv_layer(input_)


# 2D Transposed Convolution
def deconv2d(
    input_,
    output_shape,
    k_h=5,
    k_w=5,
    d_h=2,
    d_w=2,
    stddev=0.02,
    name="deconv2d",
    with_w=False,
):
    initializer = tf.random_normal_initializer(stddev=stddev)
    deconv_layer = tf.keras.layers.Conv2DTranspose(
        filters=output_shape[-1],
        kernel_size=(k_h, k_w),
        strides=(d_h, d_w),
        padding="same",
        kernel_initializer=initializer,
        bias_initializer=tf.zeros_initializer(),
        name=name,
    )
    result = deconv_layer(input_)
    if with_w:
        return result, deconv_layer.kernel, deconv_layer.bias
    return result


# Leaky ReLU
def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


# Fully connected linear layer
def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    initializer = tf.random_normal_initializer(stddev=stddev)
    bias_initializer = tf.constant_initializer(bias_start)
    dense_layer = tf.keras.layers.Dense(
        units=output_size,
        kernel_initializer=initializer,
        bias_initializer=bias_initializer,
        name=scope or "Linear",
    )
    output = dense_layer(input_)
    if with_w:
        return output, dense_layer.kernel, dense_layer.bias
    return output
