# https://github.com/sujaybabruwad/LeNet-in-Tensorflow/blob/master/LeNet-Lab.ipynb
# https://www.kaggle.com/niranjanjagannath/lenet-5-architecture-for-mnist-using-tensorflow

import tensorflow as tf
from tensorflow.layers import conv1d, conv2d, max_pooling1d, max_pooling2d, flatten, dense, average_pooling2d, average_pooling1d


def initializers(bias_init, l2str):
    weight_init = tf.initializers.variance_scaling(scale=2.0,
                                                   # He initialization https://arxiv.org/pdf/1502.01852v1.pdf
                                                   mode='fan_in',
                                                   distribution='normal'
                                                   )
    bias_ini = tf.constant_initializer(value=bias_init)  # small constant
    regularizer = tf.contrib.layers.l2_regularizer(scale=l2str)
    return weight_init, bias_ini, regularizer


def lenet5_1d(x, keep_prob, is_train, conf):
    """modified LeNet5 (ReLU instead sigmoid, dropout after FC layers)"""
    def _dense(inputs, neurons, name, activation=tf.nn.relu):
        """wrapper function for dense layer"""
        w_ini, b_ini, r_ini = initializers(conf.bias_init, conf.l2_str)
        return dense(inputs, neurons, activation=activation, bias_initializer=b_ini,
                     kernel_initializer=w_ini,
                     kernel_regularizer=r_ini,
                     name=name)

    def _conv1d(inputs, n_filters, size, name, padding='VALID'):
        """wrapper function for 1D convolutional layer"""
        w_ini, b_ini, r_ini = initializers(conf.bias_init, conf.l2_str)
        return conv1d(inputs, n_filters, size, padding=padding,
                      activation=tf.nn.relu,
                      bias_initializer=b_ini,
                      kernel_initializer=w_ini,
                      kernel_regularizer=r_ini,
                      name=name)
    # split signals according to length and channels
    x_multichannel = tf.reshape(x, [-1, conf.seq_lngth, conf.num_ch])
    # inference
    net = _conv1d(x_multichannel, 6, 5, 'conv1', padding='SAME')
    net = max_pooling1d(net, 2, 2, name='pool1')
    net = _conv1d(net, 16, 5, 'conv2')
    net = max_pooling1d(net, 2, 2, name='pool2')
    net = flatten(net, 'flat_layer')
    net = _dense(net, 120, 'fc1')
    net = tf.nn.dropout(net, keep_prob, name='dropout1')
    net = _dense(net, 84, 'fc2')
    net = tf.nn.dropout(net, keep_prob, name='dropout2')
    # output layer (softmax applied at loss function)
    logits = _dense(net, conf.num_cl, activation=None, name='logits')
    return logits


def lenet5_2d(x, keep_prob, is_train, conf):
    """modified LeNet5 (ReLU instead sigmoid, dropout after FC layers)"""
    def _dense(inputs, neurons, name, activation=tf.nn.relu):
        """wrapper function for dense layer"""
        w_ini, b_ini, r_ini = initializers(conf.bias_init, conf.l2_str)
        return dense(inputs, neurons, activation=activation, bias_initializer=b_ini,
                     kernel_initializer=w_ini,
                     kernel_regularizer=r_ini,
                     name=name)

    def _conv2d(inputs, n_filters, size, name, padding='VALID'):
        """wrapper function for 2D convolutional layer"""
        w_ini, b_ini, r_ini = initializers(conf.bias_init, conf.l2_str)
        return conv2d(inputs, n_filters, size, padding=padding,
                      activation=tf.nn.relu,
                      bias_initializer=b_ini,
                      kernel_initializer=w_ini,
                      kernel_regularizer=r_ini,
                      name=name)
    # split signals according to length and channels
    x_multichannel = tf.reshape(x, [-1, conf.img_dim[0], conf.img_dim[1], conf.num_ch])
    # inference
    net = _conv2d(x_multichannel, 6, 5, 'conv1', padding='SAME')
    net = max_pooling2d(net, [2, 2], 2, name='pool1')
    net = _conv2d(net, 16, 5, 'conv2')
    net = max_pooling2d(net, [2, 2], 2, name='pool2')
    net = flatten(net, 'flat_layer')
    net = _dense(net, 120, 'fc1')
    net = tf.nn.dropout(net, keep_prob, name='dropout1')
    net = _dense(net, 84, 'fc2')
    net = tf.nn.dropout(net, keep_prob, name='dropout2')
    # output layer (softmax applied at loss function)
    logits = _dense(net, conf.num_cl, activation=None, name='logits')
    return logits


