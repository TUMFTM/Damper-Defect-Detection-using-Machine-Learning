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


# Implementation influences: https://github.com/vonclites/squeezenet/ , https://github.com/rcmalli/keras-squeezenet
# Original SqueezeNet paper: https://arxiv.org/abs/1602.07360

import tensorflow as tf
from tensorflow.layers import conv1d, conv2d, max_pooling1d, max_pooling2d


def squeezenet1d(x, keep_prob, is_train, conf):

    def initializers(bias_init=conf.bias_init, l2str=conf.l2_str):
        weight_init = tf.initializers.variance_scaling(scale=2.0,
                                                       # He initialization https://arxiv.org/pdf/1502.01852v1.pdf
                                                       mode='fan_in',
                                                       distribution='normal'
                                                       )
        bias_ini = tf.constant_initializer(value=bias_init)  # small constant
        regularizer = tf.contrib.layers.l2_regularizer(scale=l2str)
        return weight_init, bias_ini, regularizer

    def _squeeze1d(inputs, n_outputs):
        w_ini, b_ini, r_ini = initializers()
        return conv1d(inputs=inputs, filters=n_outputs,
                      kernel_size=1, strides=1, name='squeeze', activation=tf.nn.relu,
                      kernel_initializer=w_ini, bias_initializer=b_ini,
                      kernel_regularizer=r_ini
                      )

    def _expand1d(inputs, n_outputs):
        w_ini, b_ini, r_ini = initializers()
        with tf.variable_scope('expand'):
            e1x1 = conv1d(inputs=inputs, filters=n_outputs,
                          kernel_size=1, strides=1, name='1x1', activation=tf.nn.relu,
                          kernel_initializer=w_ini, bias_initializer=b_ini,
                          kernel_regularizer=r_ini
                          )
            e3x3 = conv1d(inputs=inputs, filters=n_outputs, padding='SAME',
                          kernel_size=3, strides=1, name='3x3', activation=tf.nn.relu,
                          kernel_initializer=w_ini, bias_initializer=b_ini,
                          kernel_regularizer=r_ini
                          )
        return tf.concat([e1x1, e3x3], -1)

    def fire_module1d(inputs,
                      squeeze_depth,
                      expand_depth,
                      name=None):
        with tf.variable_scope(name, 'fire', values=[inputs]):
            layer = _squeeze1d(inputs, squeeze_depth)
            layer = _expand1d(layer, expand_depth)
        return layer

    def _squeezenet1d(flat_input, keep_prob, n_classes):
        w_ini, b_ini, r_ini = initializers()
        x_multichannel = tf.reshape(flat_input, [-1, conf.seq_lngth, conf.num_ch])
        net = conv1d(x_multichannel, filters=96, kernel_size=7, name='conv1',
                     kernel_initializer=w_ini, bias_initializer=b_ini,
                     kernel_regularizer=r_ini)
        net = max_pooling1d(net, pool_size=3, strides=2, name='maxpool1')
        net = fire_module1d(net, 16, 64, name='fire2')
        net = fire_module1d(net, 16, 64, name='fire3')
        net = fire_module1d(net, 32, 128, name='fire4')
        net = max_pooling1d(net, pool_size=3, strides=2, name='maxpool4')
        net = fire_module1d(net, 32, 128, name='fire5')
        net = fire_module1d(net, 48, 192, name='fire6')
        net = fire_module1d(net, 48, 192, name='fire7')
        net = fire_module1d(net, 64, 256, name='fire8')
        net = max_pooling1d(net, pool_size=3, strides=2, name='maxpool8')
        net = fire_module1d(net, 64, 256, name='fire9')
        net = tf.nn.dropout(net, keep_prob=keep_prob, name='dropout9')
        net = conv1d(net, n_classes, 1, 1, name='conv10',
                     kernel_initializer=w_ini, bias_initializer=b_ini,
                     kernel_regularizer=r_ini)
        logits = tf.reduce_mean(net, axis=1, name='global_avgpool10')   # global average pooling
        return logits

    return _squeezenet1d(x, keep_prob, conf.num_cl)


def squeezenet2d(x, keep_prob, is_train, conf):

    def initializers(bias_init=conf.bias_init, l2str=conf.l2_str):
        weight_init = tf.initializers.variance_scaling(scale=2.0,
                                                       # He initialization https://arxiv.org/pdf/1502.01852v1.pdf
                                                       mode='fan_in',
                                                       distribution='normal'
                                                       )
        bias_ini = tf.constant_initializer(value=bias_init)  # small constant
        regularizer = tf.contrib.layers.l2_regularizer(scale=l2str)
        return weight_init, bias_ini, regularizer

    def _squeeze2d(inputs, n_outputs):
        w_ini, b_ini, r_ini = initializers()
        return conv2d(inputs=inputs, filters=n_outputs,
                      kernel_size=1, strides=1, name='squeeze', activation=tf.nn.relu,
                      kernel_initializer=w_ini, bias_initializer=b_ini,
                      kernel_regularizer=r_ini
                      )

    def _expand2d(inputs, n_outputs):
        w_ini, b_ini, r_ini = initializers()
        with tf.variable_scope('expand'):
            e1x1 = conv2d(inputs=inputs, filters=n_outputs,
                          kernel_size=1, strides=1, name='1x1', activation=tf.nn.relu,
                          kernel_initializer=w_ini, bias_initializer=b_ini,
                          kernel_regularizer=r_ini
                          )
            e3x3 = conv2d(inputs=inputs, filters=n_outputs, padding='SAME',
                          kernel_size=3, strides=1, name='3x3', activation=tf.nn.relu,
                          kernel_initializer=w_ini, bias_initializer=b_ini,
                          kernel_regularizer=r_ini
                          )
        return tf.concat([e1x1, e3x3], -1)

    def fire_module2d(inputs,
                      squeeze_depth,
                      expand_depth,
                      name=None):
        with tf.variable_scope(name, 'fire', values=[inputs]):
            layer = _squeeze2d(inputs, squeeze_depth)
            layer = _expand2d(layer, expand_depth)
        return layer

    def _squeezenet2d(flat_input, keep_prob, n_classes):
        w_ini, b_ini, r_ini = initializers()
        x_multichannel = tf.reshape(flat_input, [-1, conf.img_dim[0], conf.img_dim[1], conf.num_ch])
        net = conv2d(x_multichannel, filters=96, kernel_size=7, name='conv1',   # keras SqueezeNet uses 3x3 kernel
                     kernel_initializer=w_ini, bias_initializer=b_ini,
                     kernel_regularizer=r_ini)
        net = max_pooling2d(net, pool_size=3, strides=2, name='maxpool1')
        net = fire_module2d(net, 16, 64, name='fire2')
        net = fire_module2d(net, 16, 64, name='fire3')
        net = fire_module2d(net, 32, 128, name='fire4')
        net = max_pooling2d(net, pool_size=3, strides=2, name='maxpool4')
        net = fire_module2d(net, 32, 128, name='fire5')
        net = fire_module2d(net, 48, 192, name='fire6')
        net = fire_module2d(net, 48, 192, name='fire7')
        net = fire_module2d(net, 64, 256, name='fire8')
        net = max_pooling2d(net, pool_size=3, strides=2, name='maxpool8')
        net = fire_module2d(net, 64, 256, name='fire9')
        net = tf.nn.dropout(net, keep_prob=keep_prob, name='dropout9')
        net = conv2d(net, n_classes, 1, 1, name='conv10',
                     kernel_initializer=w_ini, bias_initializer=b_ini,
                     kernel_regularizer=r_ini)
        logits = tf.reduce_mean(net, axis=[1, 2], name='global_avgpool10')
        return logits

    return _squeezenet2d(x, keep_prob, conf.num_cl)
