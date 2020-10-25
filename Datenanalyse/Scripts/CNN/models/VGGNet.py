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


def vgg_11_1d(x, keep_proba, conf):

    def _dense(inputs, neurons, name, activation=tf.nn.relu, use_bias=True):
        """wrapper function for dense layer"""
        w_ini, b_ini, r_ini = initializers(conf.bias_init, conf.l2_str)
        return dense(inputs, neurons, activation=activation, bias_initializer=b_ini,
                     kernel_initializer=w_ini,
                     kernel_regularizer=r_ini,
                     name=name,
                     use_bias=use_bias)

    def _conv1d(inputs, n_filters, size, name, padding='SAME'):
        """wrapper function for 1D convolutional layer"""
        w_ini, b_ini, r_ini = initializers(conf.bias_init, conf.l2_str)
        return conv1d(inputs, n_filters, size, padding=padding,
                      activation=tf.nn.relu,
                      bias_initializer=b_ini,
                      kernel_initializer=w_ini,
                      kernel_regularizer=r_ini,
                      name=name)

    # https://arxiv.org/abs/1409.1556 --> configuration A
    w_ini, b_ini, r_ini = initializers(conf.bias_init, conf.l2_str)
    x_multichannel = tf.reshape(x, [-1, conf.seq_lngth, conf.num_ch])
    net = _conv1d(x_multichannel, 64, 3, 'conv1')
    net = max_pooling1d(net, 2, 2, 'same', name='maxpool1')
    net = _conv1d(net, 128, 3, 'conv2')
    net = max_pooling1d(net, 2, 2, 'same', name='maxpool2')
    net = _conv1d(net, 256, 3, 'conv3_1')
    net = _conv1d(net, 256, 3, 'conv3_2')
    net = max_pooling1d(net, 2, 2, 'same', name='maxpool3')
    net = _conv1d(net, 512, 3, 'conv4_1')
    net = _conv1d(net, 512, 3, 'conv4_2')
    net = max_pooling1d(net, 2, 2, 'same', name='maxpool4')
    net = _conv1d(net, 512, 3, 'conv5_1')
    net = _conv1d(net, 512, 3, 'conv5_2')
    net = max_pooling1d(net, 2, 2, 'same', name='maxpool5')

    net = flatten(net, name='flat_layer')
    net = _dense(net, 4096, 'fc6')
    net = tf.nn.dropout(net, keep_proba, name='dropout6')
    net = _dense(net, 4096, 'fc7')
    net = tf.nn.dropout(net, keep_proba, name='dropout7')

    logits = _dense(net, conf.num_cl, 'logits', activation=None)
    # softmax performed at loss function
    return logits
