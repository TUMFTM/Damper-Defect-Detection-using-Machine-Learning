# https://github.com/taki0112/SENet-Tensorflow/blob/master/SE_ResNeXt.py

# MIT License
#
# Copyright (c) 2020 Thomas Zehelein and Thomas Hemmert-Pottmann
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow as tf
import numpy as np


# helper/wrapper functions
def _conv_1d(inputs, filters, kernel, stride, padding='SAME', layer_name="conv"):
    network = tf.layers.conv1d(inputs=inputs, use_bias=False, filters=filters, kernel_size=kernel, strides=stride, padding=padding, name=layer_name)
    return network


def _conv_2d(inputs, filters, kernel, stride, padding='SAME', layer_name="conv"):
    network = tf.layers.conv2d(inputs=inputs, use_bias=False, filters=filters, kernel_size=kernel, strides=stride, padding=padding, name=layer_name)
    return network


def _global_average_pooling_1d(x):
    return tf.reduce_mean(x, axis=1, name='Global_avg_pooling')


def _global_average_pooling_2d(x):
    return tf.reduce_mean(x, axis=[1, 2], name='Global_avg_pooling')


def _average_pooling_1d(x, pool_size=2, stride=2, padding='SAME'):
    return tf.layers.average_pooling1d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def _average_pooling_2d(x, pool_size=2, stride=2, padding='SAME'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


# https://github.com/hujie-frank/SENet, r=16 for all models in SENet-Paper
def _squeeze_excitation_layer1d(input_x, out_dim, layer_name, ratio=16):
    with tf.variable_scope(layer_name):
        squeeze = _global_average_pooling_1d(input_x)
        excitation = tf.layers.dense(squeeze, units=out_dim // ratio, activation=tf.nn.relu, name= layer_name + '_fc1')
        excitation = tf.layers.dense(excitation, units=out_dim, activation=tf.nn.sigmoid, name=layer_name + '_fc2')
        excitation = tf.reshape(excitation, [-1, 1, out_dim])
        scale = input_x * excitation
        return scale


def _squeeze_excitation_layer2d(input_x, out_dim, layer_name, ratio=16):
    with tf.variable_scope(layer_name):
        squeeze = _global_average_pooling_2d(input_x)
        excitation = tf.layers.dense(squeeze, units=out_dim // ratio, activation=tf.nn.relu, name= layer_name + '_fc')
        excitation = tf.layers.dense(excitation, units=out_dim, activation=tf.nn.sigmoid, name=layer_name + '_fc2')
        excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
        scale = input_x * excitation
        return scale


# 1D SE_ResNeXt
def se_resnext_1d(inputs, train_phase, conf):
    # 16x4d template = 64 filters
    cardinality = 16  # how many split ?
    depth = 4  # depth (no. of filters) of conv layers in resnext branches

    def _first_layer(x, scope):
        with tf.variable_scope(scope):
            x = _conv_1d(x, filters=64, kernel=7, stride=2, layer_name='conv_3x1')
            x = tf.layers.batch_normalization(x, training=train_phase, name='bn1')
            x = tf.nn.relu(x)
            return x

    def _transform_layer(x, stride, scope):
        with tf.variable_scope(scope):
            x = tf.layers.batch_normalization(x, training=train_phase, name='bn1')
            x = tf.nn.relu(x)
            x = _conv_1d(x, filters=depth, kernel=1, stride=1, layer_name='conv_1x1')

            x = tf.layers.batch_normalization(x, training=train_phase, name='bn2')
            x = tf.nn.relu(x)
            x = _conv_1d(x, filters=depth, kernel=3, stride=stride, layer_name='conv_3x1')
            return x

    def _transition_layer(x, out_dim, scope):
        with tf.variable_scope(scope):
            x = _conv_1d(x, filters=out_dim, kernel=1, stride=1, layer_name='conv_1x1')
            x = tf.layers.batch_normalization(x, training=train_phase, name='bn1')
            x = tf.nn.relu(x)   # ReLU applied after each BN layer (resnext paper)
            return x

    def _split_layer(input_x, stride, layer_name):
        with tf.variable_scope(layer_name):
            layers_split = list()
            for i in range(cardinality):
                splits = _transform_layer(input_x, stride=stride, scope='branch_' + str(i))
                layers_split.append(splits)
            return tf.concat(layers_split, axis=2)

    def _residual_layer(input_x, out_dim, layer_num, res_block, first_stage=False):
        # split + transform(bottleneck) + transition + merge
        with tf.variable_scope('Resblock_' + layer_num):
            for i in range(res_block):
                input_dim = int(np.shape(input_x)[-1])

                if input_dim * 2 == out_dim:
                    flag = True
                    stride = 1 if first_stage else 2  # stride of 2 for downsampling in first 3x3 conv of each stage (except 1st stage)
                    channel = input_dim // 2
                else:
                    flag = False
                    stride = 1

                x = _split_layer(input_x, stride=stride, layer_name='split_' + layer_num + '_' + str(i))
                x = _transition_layer(x, out_dim=out_dim, scope='trans_' + layer_num + '_' + str(i))  # "early concatenation" resnext paper
                x = _squeeze_excitation_layer1d(x, out_dim=out_dim, layer_name='se_' + layer_num + '_' + str(i))

                if flag is True:
                    if first_stage:
                        pad_input_x = input_x
                    else:
                        pad_input_x = _average_pooling_1d(input_x)
                    pad_input_x = tf.pad(pad_input_x, [[0, 0], [0, 0], [channel, channel]])  # [?, width, channel]
                else:
                    pad_input_x = input_x

                input_x = x + pad_input_x

        return input_x

    x_multichannel = tf.reshape(inputs, [-1, conf.seq_lngth, conf.num_ch])
    net = _first_layer(x_multichannel, scope='first_layer')
    net = tf.layers.max_pooling1d(net, pool_size=3, strides=2)
    # SE-ResNeXt-29
    # double filters after each stage, use double the size of template (16x4)
    out_depth = 2*cardinality*depth
    net = _residual_layer(net, out_dim=out_depth, layer_num='1', res_block=3, first_stage=True)
    net = _residual_layer(net, out_dim=out_depth*2, layer_num='2', res_block=3)
    net = _residual_layer(net, out_dim=out_depth*4, layer_num='3', res_block=3)

    net = _global_average_pooling_1d(net)
    net = tf.layers.flatten(net)

    logits = tf.layers.dense(net, units=conf.num_cl, name='last_fc')
    return logits


# 2D SE_ResNeXt
def se_resnext_2d(inputs, train_phase, conf):
    # SE-ResNeXt with full pre-activation (SE paper uses original activation and relu after shortcut connection)
    cardinality = 16  # how many split ? # number of parallel resnext branches --> refers to inception style
    depth = 4  # depth of conv layers in resnext branches --> "32x4d" template

    def _first_layer(x, scope):
        with tf.variable_scope(scope):
            # x = _conv_2d(x, filters=64, kernel=5, stride=1, layer_name='conv_3x3')
            x = _conv_2d(x, filters=64, kernel=7, stride=2, layer_name='conv_3x3')  # TODO decrease to 3x3 maybe (authors did for CIFAR-100, too)
            x = tf.layers.batch_normalization(x, training=train_phase, name='bn1')
            x = tf.nn.relu(x)
            return x

    def _transform_layer(x, stride, scope):
        with tf.variable_scope(scope):  # modified order for full pre-activation --> paper
            x = tf.layers.batch_normalization(x, training=train_phase, name='bn1')
            x = tf.nn.relu(x)
            x = _conv_2d(x, filters=depth, kernel=1, stride=1, layer_name='conv_1x1')

            x = tf.layers.batch_normalization(x, training=train_phase, name='bn2')
            x = tf.nn.relu(x)
            x = _conv_2d(x, filters=depth, kernel=3, stride=stride, layer_name='conv_3x3')
            return x

    def _transition_layer(x, out_dim, scope):
        with tf.variable_scope(scope):
            x = _conv_2d(x, filters=out_dim, kernel=1, stride=1, layer_name='conv_1x1')
            x = tf.layers.batch_normalization(x, training=train_phase, name='bn1')
            x = tf.nn.relu(x)   # ReLU applied after each BN layer (resnext paper)
            return x

    def _split_layer(input_x, stride, layer_name):
        with tf.variable_scope(layer_name):
            layers_split = list()
            for i in range(cardinality):
                splits = _transform_layer(input_x, stride=stride, scope='branch_' + str(i))
                layers_split.append(splits)
            return tf.concat(layers_split, axis=3)

    def _residual_layer(input_x, out_dim, layer_num, res_block, first_stage=False):
        # split + transform(bottleneck) + transition + merge
        with tf.variable_scope('Resblock_' + layer_num):
            for i in range(res_block):
                input_dim = int(np.shape(input_x)[-1])

                if input_dim * 2 == out_dim:
                    flag = True
                    stride = 1 if first_stage else 2  # stride of 2 for downsampling in first 3x3 conv of each stage (except 1st stage)
                    channel = input_dim // 2
                else:
                    flag = False
                    stride = 1

                x = _split_layer(input_x, stride=stride, layer_name='split_' + layer_num + '_' + str(i))
                x = _transition_layer(x, out_dim=out_dim, scope='trans_' + layer_num + '_' + str(i))  # "early concatenation" resnext paper
                x = _squeeze_excitation_layer2d(x, out_dim=out_dim, layer_name='se_' + layer_num + '_' + str(i))

                if flag is True:
                    if first_stage:
                        pad_input_x = input_x
                    else:
                        pad_input_x = _average_pooling_2d(input_x)
                    pad_input_x = tf.pad(pad_input_x, [[0, 0], [0, 0], [0, 0], [channel, channel]])  # [?, height, width, channel]
                else:
                    pad_input_x = input_x

                input_x = x + pad_input_x

            return input_x

    x_multichannel = tf.reshape(inputs, [-1, conf.img_dim[0], conf.img_dim[1], conf.num_ch])
    net = _first_layer(x_multichannel, scope='first_layer')
    net = tf.layers.max_pooling2d(net, pool_size=3, strides=2)    # TODO check if required
    # res_block parameter defines how many consecutive SE-ResNeXt blocks should be there
    out_depth = 2*cardinality*depth
    net = _residual_layer(net, out_dim=out_depth, layer_num='1', res_block=3, first_stage=True)
    net = _residual_layer(net, out_dim=out_depth*2, layer_num='2', res_block=3)
    net = _residual_layer(net, out_dim=out_depth*4, layer_num='3', res_block=3)
    net = _global_average_pooling_2d(net)
    net = tf.layers.flatten(net)
    logits = tf.layers.dense(net, units=conf.num_cl, name='last_fc')

    return logits
