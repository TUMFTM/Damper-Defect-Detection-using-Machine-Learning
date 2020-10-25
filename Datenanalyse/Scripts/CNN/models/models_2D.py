# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/vgg.py

import tensorflow as tf
from tensorflow.layers import conv2d, max_pooling2d, dense, flatten
from models.LeNet import lenet5_2d
from models.SqueezeNet import squeezenet2d
from models.SE_ResNeXt import se_resnext_2d
from models.DamNet import damnet_v1
from models.Baseline_new import  baseline_new
from models.Paper import paper_nets
import numpy as np


def conv_net(x, keep_prob, train_phase, conf):
    def initializers(bias_init=conf.bias_init, l2str=conf.l2_str):
        weight_init = tf.initializers.variance_scaling(scale=2.0,
                                                       # He initialization https://arxiv.org/pdf/1502.01852v1.pdf
                                                       mode='fan_in',
                                                       distribution='normal'
                                                       )
        bias_ini = tf.constant_initializer(value=bias_init)  # small constant
        regularizer = tf.contrib.layers.l2_regularizer(scale=l2str)
        return weight_init, bias_ini, regularizer

    def _dense(inputs, neurons, name, activation=tf.nn.relu, use_bias=True):
        """wrapper function for dense layer"""
        w_ini, b_ini, r_ini = initializers(conf.bias_init, conf.l2_str)
        return dense(inputs, neurons, activation=activation, bias_initializer=b_ini,
                     kernel_initializer=w_ini,
                     kernel_regularizer=r_ini,
                     name=name,
                     use_bias=use_bias)

    def _conv2d(inputs, n_filters, size, name, padding='SAME', use_bias=True):
        """wrapper function for 2D convolutional layer"""
        w_ini, b_ini, r_ini = initializers(conf.bias_init, conf.l2_str)
        return conv2d(inputs, n_filters, size, padding=padding,
                      activation=tf.nn.relu,
                      bias_initializer=b_ini,
                      kernel_initializer=w_ini,
                      kernel_regularizer=r_ini,
                      use_bias=use_bias,
                      name=name)

    def grouped_sepconv_layer(x_multich, splits, num_filters, filter_size):
        w_ini, b_ini, reg_ini = initializers()
        ch_multiplier = 32
        if np.sum(splits, axis=-1) != x_multich.shape[-1]:
            raise ValueError('Sum of specified channels for splitting does not match incoming tensor')
        with tf.variable_scope('DamLayer'):
            x_split = tf.split(x_multich, splits, axis=len(x_multich.shape)-1)
            layers_split = []
            filter_sum = 0
            for i in range(len(splits)):
                filters = splits[i]*ch_multiplier
                split_layer = tf.layers.conv2d(x_split[i], filters, filter_size, padding='SAME', activation=tf.nn.relu,
                                               kernel_initializer=w_ini, kernel_regularizer=reg_ini, bias_initializer=b_ini)
                layers_split.append(split_layer)
                filter_sum += filters
            layer = tf.concat(layers_split, axis=-1)
            layer = tf.layers.conv2d(layer, num_filters, 1, activation=tf.nn.relu, padding='SAME', bias_initializer=b_ini,
                                     kernel_initializer=w_ini, kernel_regularizer=reg_ini,name='conv_1x1')
        return layer

    def inference_baseline(x, keep_prob, train_phase, n_classes):
        x_multichannel = tf.reshape(x, [-1, conf.img_dim[0], conf.img_dim[1], conf.num_ch])

        net = _conv2d(x_multichannel, 64, 9, 'conv1')
        net = tf.layers.max_pooling2d(net, 2, 2, name='maxpool1')
        net = tf.layers.flatten(net, 'flat_layer')
        net = _dense(net, 128, 'fc1')
        # apply dropout to first FC layer
        net = tf.nn.dropout(net, keep_prob, name='Dropout')
        # 2nd fully connected layer = output
        logits = _dense(net, n_classes, 'output', activation=None)
        return logits

    def doubleconv(x, keep_proba, is_train, n_classes):
        # Architecture see Verstraete_2017
        x_multichannel = tf.reshape(x, [-1, conf.img_dim[0], conf.img_dim[1], conf.num_ch])
        with tf.variable_scope("Block1"):
            # net = tf.layers.batch_normalization(x_multichannel, training=is_train, renorm=True)
            net = _conv2d(x_multichannel, 3, 32, "conv1_relu", use_bias=True)
            net = _conv2d(net, 3, 32, "conv2_relu")
            net = tf.layers.max_pooling2d(net, 2, 2, name="maxpool1")
        with tf.variable_scope("Block2"):
            # net = tf.layers.batch_normalization(net, training=is_train, renorm=True)
            net = _conv2d(net, 3, 64, "conv1_relu", use_bias=True)
            net = _conv2d(net, 3, 64, "conv2_relu")
            net = tf.layers.max_pooling2d(net, 2, 2, name="maxpool2")
        with tf.variable_scope("Block3"):
            # net = tf.layers.batch_normalization(net, training=is_train)
            net = _conv2d(net, 3, 128, "conv1_relu", use_bias=True)
            net = _conv2d(net, 3, 128, "conv2_relu")
            net = tf.layers.max_pooling2d(net, 2, 2, name="maxpool3")
        with tf.variable_scope("FC-Block"):
            net = tf.layers.flatten(net, name="flat_layer")
            # net = tf.layers.batch_normalization(net, training=is_train, renorm=True)
            net = _dense(net, 100, 'fc1_relu', use_bias=True)
            # apply dropout to first FC layer
            net = tf.nn.dropout(net, keep_proba, name="fc1_drop")
            net = _dense(net, 100, 'fc2_relu', use_bias=True)
            # # apply dropout to 2nd FC layer
            net = tf.nn.dropout(net, keep_proba, name="fc2_drop")
            logits = _dense(net, n_classes, 'Logits', activation=None)
        return logits

    def sepconv(x, keep_proba, is_train, num_classes):
        w_ini, b_ini, reg_ini = initializers()
        x_multichannel = tf.reshape(x, [-1, conf.img_dim[0], conf.img_dim[1], conf.num_ch])
        net = tf.layers.separable_conv2d(x_multichannel, 64, 9, padding='SAME', activation=tf.nn.relu,
                                         depth_multiplier=32,
                                         depthwise_initializer=w_ini, pointwise_initializer=w_ini,
                                         bias_initializer=b_ini,
                                         depthwise_regularizer=reg_ini, pointwise_regularizer=reg_ini)
        net = tf.layers.max_pooling2d(net, 2, 2, name='maxpool1')
        net = tf.layers.flatten(net, 'flat_layer')
        net = tf.layers.dense(net, 128, activation=tf.nn.relu, name="fc1_relu_bn",
                              bias_initializer=b_ini, kernel_initializer=w_ini,
                              kernel_regularizer=reg_ini)
        # apply dropout to first FC layer
        net = tf.nn.dropout(net, keep_proba, name='Dropout')
        # 2nd fully connected layer = output
        logits = tf.layers.dense(inputs=net, units=num_classes, activation=None, kernel_initializer=w_ini,
                                 bias_initializer=b_ini, kernel_regularizer=reg_ini, name='Output')
        return logits

    def grouped_sepconv(x, keep_proba, is_train, num_classes, splits):
        w_ini, b_ini, reg_ini = initializers()
        # splits = [1,1,1,1,1,1]  # specify the grouping of channels (currently they have to be next to each other)
        x_multichannel = tf.reshape(x, [-1, conf.img_dim[0], conf.img_dim[1], conf.num_ch])
        net = grouped_sepconv_layer(x_multichannel, splits, 64, 9)
        net = tf.layers.max_pooling2d(net, 2, 2, name='maxpool1')
        net = tf.layers.flatten(net, 'flat_layer')
        net = tf.layers.dense(net, 128, activation=tf.nn.relu, name="fc1_relu_bn",
                              bias_initializer=b_ini, kernel_initializer=w_ini,
                              kernel_regularizer=reg_ini)
        # apply dropout to first FC layer
        net = tf.nn.dropout(net, keep_proba, name='Dropout')
        # 2nd fully connected layer = output
        logits = tf.layers.dense(inputs=net, units=num_classes, activation=None, kernel_initializer=w_ini,
                                 bias_initializer=b_ini, kernel_regularizer=reg_ini, name='Output')
        return logits

    def ticnn(x, keep_proba, is_train, num_classes):
        # Zhang, Wei; Li, Chuanhao; Peng, Gaoliang; Chen, Yuanhang; Zhang, Zhujun
        # A deep convolutional neural network with new training methods for
        # bearing fault diagnosis under noisy environment and different working load
        # DOI 10.1016/j.ymssp.2017.06.022
        # see section 4.2 and table 2 (paper) for architectural details
        # THP_MOD: original paper uses seq_length of 2048, filter_size=64, stride=8,
        #  paper applies BN before activation
        w_ini, b_ini, r_ini = initializers(conf.bias_init, conf.l2_str)
        x_multichannel = tf.reshape(x, [-1, conf.img_dim[0], conf.img_dim[1], conf.num_ch])
        net = _conv2d(x_multichannel, 16, 4, 'Convolution1', use_bias=False)
        net = tf.layers.batch_normalization(net, training=is_train, gamma_regularizer=r_ini)
        net = tf.nn.dropout(net, keep_proba, name="Kernel_dropout")
        net = tf.layers.max_pooling2d(net, 2, 2, name="Pooling1")
        net = _conv2d(net, 32, 3, "Convolution2", use_bias=False)  # "same" padding, stride of 1, use relu
        net = tf.layers.batch_normalization(net, training=is_train, gamma_regularizer=r_ini)
        net = tf.layers.max_pooling2d(net, 2, 2, name="Pooling2")
        net = _conv2d(net, 64, 3, "Convolution3", use_bias=False)  # "same" padding, stride of 1, use bn, use relu
        net = tf.layers.batch_normalization(net, training=is_train, gamma_regularizer=r_ini)
        net = tf.layers.max_pooling2d(net, 2, 2, name="Pooling3")
        ### THP_MOD: had to comment out due to negative dimension after pooling
        # net = _conv2d(net, 64, 3, "Convolution4", use_bias=False)  # "same" padding, stride of 1, use bn, use relu
        # net = tf.layers.batch_normalization(net, training=is_train, gamma_regularizer=r_ini)
        # net = tf.layers.max_pooling2d(net, 2, 2, name="Pooling4")
        # net = new_conv_layer(net, 64, 3, is_train, "Convolution5")  # "same" padding, stride of 1, use bn, use relu
        # net = tf.layers.max_pooling1d(net, 2, 2, name="Pooling5")
        net = _conv2d(net, 64, 3, "Convolution6", padding="VALID", use_bias=False)  # no padding, stride of 1, use bn, use relu
        net = tf.layers.batch_normalization(net, training=is_train, gamma_regularizer=r_ini)
        net = tf.layers.max_pooling2d(net, 2, 2, name="Pooling6")
        net = tf.layers.flatten(net,"flat_layer")
        net = tf.layers.dense(net, 100, name="Fully-connected", use_bias=False, activation=tf.nn.relu,
                              kernel_regularizer=r_ini, bias_initializer=b_ini, kernel_initializer=w_ini)
        net = tf.layers.batch_normalization(net, training=is_train, gamma_regularizer=r_ini)
        # output layer (softmax applied at loss function)
        logits = tf.layers.dense(net, num_classes, activation=None, name='logits')

        return logits
    def vgg_11(x, keep_proba, is_train, num_classes):
        # https://arxiv.org/abs/1409.1556 --> configuration A
        w_ini, b_ini, r_ini = initializers(conf.bias_init, conf.l2_str)
        x_multichannel = tf.reshape(x, [-1, conf.img_dim[0], conf.img_dim[1], conf.num_ch])
        net = _conv2d(x_multichannel, 64, 3, 'conv1')
        net = max_pooling2d(net, 2, 2, 'same', name='maxpool1')
        net = _conv2d(net, 128, 3, 'conv2')
        net = max_pooling2d(net, 2, 2, 'same', name='maxpool2')
        net = _conv2d(net, 256, 3, 'conv3_1')
        net = _conv2d(net, 256, 3, 'conv3_2')
        net = max_pooling2d(net, 2, 2, 'same', name='maxpool3')
        net = _conv2d(net, 512, 3, 'conv4_1')
        net = _conv2d(net, 512, 3, 'conv4_2')
        net = max_pooling2d(net, 2, 2, 'same', name='maxpool4')
        net = _conv2d(net, 512, 3, 'conv5_1')
        net = _conv2d(net, 512, 3, 'conv5_2')
        net = max_pooling2d(net, 2, 2, 'same', name='maxpool5')

        net = flatten(net, name='flat_layer')
        net = _dense(net, 4096, 'fc6')
        net = tf.nn.dropout(net, keep_proba, name='dropout6')
        net = _dense(net, 4096, 'fc7')
        net = tf.nn.dropout(net, keep_proba, name='dropout7')
        # # TODO consider replacing fc layers by conv layers
        # #  as in https://github.com/tensorflow/models/blob/master/research/slim/nets/vgg.py
        # #  for "if include_top" see https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg16.py
        # net = _conv2d(net, 4096, 7, 'fc6_replacement', padding='VALID')
        # net = tf.nn.dropout(net, keep_proba, name='dropout7')
        # net = _conv2d(net, 4096, 1, 'fc7_replacement')
        # net = tf.reduce_mean(net, axis=[1, 2], name='global_avg_pool8')
        logits = _dense(net, conf.num_cl, 'logits', activation=None)
        # softmax performed at loss function
        return logits

    if conf.model == 'baseline':
        out = inference_baseline(x, keep_prob, train_phase, conf.num_cl)
    elif conf.model == 'baseline_new':
        out = baseline_new(x, keep_prob, train_phase, conf)
    elif conf.model == 'VGG':
        out = vgg_11(x, keep_proba=keep_prob, is_train=train_phase, num_classes=conf.num_cl)
    elif conf.model == 'doubleconv':
        out = doubleconv(x, keep_prob, train_phase, conf.num_cl)
    elif conf.model == 'TICNN':
        out = ticnn(x, keep_prob, train_phase, conf.num_cl)
    elif conf.model == 'LeNet':
        out = lenet5_2d(x, keep_prob=keep_prob, is_train=train_phase, conf=conf)
    elif conf.model == 'SqueezeNet':
        out = squeezenet2d(x, keep_prob, train_phase, conf)
    elif conf.model == 'SE_ResNeXt':
        out = se_resnext_2d(x, train_phase, conf)
    elif conf.model == 'DamNet_v1':
        out = damnet_v1(x, keep_prob, train_phase, conf, False)
    elif conf.model == 'sepConv':
        out = sepconv(x, keep_prob, train_phase, conf.num_cl)
    elif conf.model == 'grouped_sepConv':
        out = grouped_sepconv(x, keep_prob, train_phase, conf.num_cl, [2, 2, 3])
    elif conf.model == 'paper_prep':
        out = paper_nets(x, keep_prob, conf)
    else:
        raise Warning('model parameter "%s" ill defined' % conf.model)
    return out
