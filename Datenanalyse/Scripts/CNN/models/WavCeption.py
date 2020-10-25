# https://www.kaggle.com/ivallesp/wavception-v1-a-1-d-inception-approach-lb-0-76
import tensorflow as tf


def wavception(inputs, keep_prob, train_phase, conf):
    w_initializer = tf.initializers.variance_scaling(scale=2.0,
                                                     mode='fan_in',
                                                     distribution='normal')
    b_initializer = tf.constant_initializer(value=conf.bias_init)
    regularizer = tf.contrib.layers.l2_regularizer(scale=conf.l2_str)

    def _norm_function(input, name, is_train=train_phase):
        """wrapper function for batch norm layer"""
        return tf.layers.batch_normalization(inputs=input,
                                             training=is_train,
                                             name=name,
                                             gamma_regularizer=regularizer)

    def _conv_bn_relu_layer(inputs, filters, kernel_size, nameext, padding='same', use_relu=True, use_bn=True):
        """wrapper function for convolution with batch norm (toggle bias) and relu non-linearity"""
        if use_bn:
            layer = tf.layers.conv1d(inputs=inputs,
                                     filters=filters,
                                     kernel_size=kernel_size,
                                     padding=padding,
                                     kernel_initializer=w_initializer,
                                     bias_initializer=b_initializer,
                                     kernel_regularizer=regularizer,
                                     use_bias=False,
                                     name='conv_' + nameext)
        else:
            layer = tf.layers.conv1d(inputs=inputs,
                                     filters=filters,
                                     kernel_size=kernel_size,
                                     padding=padding,
                                     kernel_initializer=w_initializer,
                                     bias_initializer=b_initializer,
                                     kernel_regularizer=regularizer,
                                     use_bias=True,
                                     name='conv_' + nameext)
        if use_bn:
            layer = _norm_function(layer, name='norm_conv_' + nameext)
        if use_relu:
            layer = tf.nn.relu(layer, name='activation_' + nameext)
        return layer

    def _inception_1d(x, depth, name, use_bn=True):
        """
        Inception 1D module implementation.
        :param x: input to the current module (4D tensor with channels-last)
        :param depth: linearly controls the depth of the network (int)
        :param name: name of the variable scope (str)
        """

        with tf.variable_scope(name):
            x_norm = _norm_function(x, 'norm_input')
            # Branch 1: 64 x conv 1x1
            branch_conv_1_1 = _conv_bn_relu_layer(inputs=x_norm, filters=16*depth, kernel_size=1, nameext='1_1')

            # Branch 2: 128 x conv 3x3
            branch_conv_3_3 = _conv_bn_relu_layer(inputs=x_norm, filters=16, kernel_size=1, nameext='3_3_1')
            branch_conv_3_3 = _conv_bn_relu_layer(inputs=branch_conv_3_3, filters=32*depth, kernel_size=3, nameext='3_3_2')

            # Branch 3: 128 x conv 5x5
            branch_conv_5_5 = _conv_bn_relu_layer(inputs=x_norm, filters=16, kernel_size=1, nameext='5_5_1')
            branch_conv_5_5 = _conv_bn_relu_layer(inputs=branch_conv_5_5, filters=32*depth, kernel_size=5, nameext='5_5_2')

            # Branch 4: 128 x conv 7x7
            branch_conv_7_7 = _conv_bn_relu_layer(inputs=x_norm, filters=16, kernel_size=1, nameext='7_7_1')
            branch_conv_7_7 = _conv_bn_relu_layer(inputs=branch_conv_7_7, filters=32*depth, kernel_size=7, nameext='7_7_2')

            # Branch 5: 16 x (max_pool 3x3 + conv 1x1)
            branch_maxpool_3_3 = tf.layers.max_pooling1d(inputs=x_norm, pool_size=3, strides=1, padding="same", name="maxpool_3")
            if use_bn:
                branch_maxpool_3_3 = _norm_function(branch_maxpool_3_3, name="norm_maxpool_3_3")
                branch_maxpool_3_3 = tf.layers.conv1d(inputs=branch_maxpool_3_3, filters=16, kernel_size=1, use_bias=False,
                                                      bias_initializer=b_initializer, kernel_initializer=w_initializer,
                                                      kernel_regularizer=regularizer,
                                                      padding="same", name="conv_maxpool_3")
            else:
                branch_maxpool_3_3 = tf.layers.conv1d(inputs=branch_maxpool_3_3, filters=16, kernel_size=1, use_bias=True,
                                                      bias_initializer=b_initializer, kernel_initializer=w_initializer,
                                                      kernel_regularizer=regularizer,
                                                      padding="same", name="conv_maxpool_3")

            # Branch 6: 16 x (max_pool 5x5 + conv 1x1)
            branch_maxpool_5_5 = tf.layers.max_pooling1d(inputs=x_norm, pool_size=5, strides=1, padding="same", name="maxpool_5")
            if use_bn:
                branch_maxpool_5_5 = _norm_function(branch_maxpool_5_5, name="norm_maxpool_5_5")
                branch_maxpool_5_5 = tf.layers.conv1d(inputs=branch_maxpool_5_5, filters=16, kernel_size=1, use_bias=False,
                                                      bias_initializer=b_initializer, kernel_initializer=w_initializer,
                                                      kernel_regularizer=regularizer,
                                                      padding="same", name="conv_maxpool_5")
            else:
                branch_maxpool_5_5 = tf.layers.conv1d(inputs=branch_maxpool_5_5, filters=16, kernel_size=1, use_bias=True,
                                                      bias_initializer=b_initializer, kernel_initializer=w_initializer,
                                                      kernel_regularizer=regularizer,
                                                      padding="same", name="conv_maxpool_5")

            # Branch 7: 16 x (avg_pool 3x3 + conv 1x1)
            branch_avgpool_3_3 = tf.layers.average_pooling1d(inputs=x_norm, pool_size=3, strides=1, padding="same", name="avgpool_3")
            if use_bn:
                branch_avgpool_3_3 = _norm_function(branch_avgpool_3_3, name="norm_avgpool_3_3")
                branch_avgpool_3_3 = tf.layers.conv1d(inputs=branch_avgpool_3_3, filters=16, kernel_size=1, use_bias=False,
                                                      bias_initializer=b_initializer, kernel_initializer=w_initializer,
                                                      kernel_regularizer=regularizer,
                                                      padding="same", name="conv_avgpool_3")
            else:
                branch_avgpool_3_3 = tf.layers.conv1d(inputs=branch_avgpool_3_3, filters=16, kernel_size=1, use_bias=True,
                                                      bias_initializer=b_initializer, kernel_initializer=w_initializer,
                                                      kernel_regularizer=regularizer,
                                                      padding="same", name="conv_avgpool_3")

            # Branch 8: 16 x (avg_pool 5x5 + conv 1x1)
            branch_avgpool_5_5 = tf.layers.average_pooling1d(inputs=x_norm, pool_size=5, strides=1, padding="same", name="avgpool_5")
            if use_bn:
                branch_avgpool_5_5 = _norm_function(branch_avgpool_5_5, name="norm_avgpool_5_5")
                branch_avgpool_5_5 = tf.layers.conv1d(inputs=branch_avgpool_5_5, filters=16, kernel_size=1, use_bias=False,
                                                      bias_initializer=b_initializer, kernel_initializer=w_initializer,
                                                      kernel_regularizer=regularizer,
                                                      padding="same", name="conv_avgpool_5")
            else:
                branch_avgpool_5_5 = tf.layers.conv1d(inputs=branch_avgpool_5_5, filters=16, kernel_size=1, use_bias=True,
                                                      bias_initializer=b_initializer, kernel_initializer=w_initializer,
                                                      kernel_regularizer=regularizer,
                                                      padding="same", name="conv_avgpool_5")

            # Concatenate
            output = tf.concat([branch_conv_1_1, branch_conv_3_3, branch_conv_5_5, branch_conv_7_7, branch_maxpool_3_3,
                               branch_maxpool_5_5, branch_avgpool_3_3, branch_avgpool_5_5], axis=-1)
            return output

    x_multichannel = tf.reshape(inputs, [-1, conf.seq_lngth, conf.num_ch])
    net = _inception_1d(x=x_multichannel, depth=1, name="Inception_1_1")
    net = _inception_1d(x=net, depth=1, name="Inception_1_2")
    net = tf.layers.max_pooling1d(net, 2, 2, name="maxpool_1")
    net = _inception_1d(x=net, depth=1, name="Inception_2_1")
    net = _inception_1d(x=net, depth=1, name="Inception_2_3")
    net = tf.layers.max_pooling1d(net, 2, 2, name="maxpool_2")
    net = _inception_1d(x=net, depth=2, name="Inception_3_1")
    net = _inception_1d(x=net, depth=2, name="Inception_3_2")
    net = tf.layers.max_pooling1d(net, 2, 2, name="maxpool_3")
    net = _inception_1d(x=net, depth=2, name="Inception_4_1")
    net = _inception_1d(x=net, depth=2, name="Inception_4_2")
    net = tf.layers.max_pooling1d(net, 2, 2, name="maxpool_4")
    net = _inception_1d(x=net, depth=3, name="Inception_5_1")
    net = _inception_1d(x=net, depth=3, name="Inception_5_2")
    net = tf.layers.max_pooling1d(net, 2, 2, name="maxpool_5")
    net = _inception_1d(x=net, depth=3, name="Inception_6_1")
    net = _inception_1d(x=net, depth=3, name="Inception_6_2")
    net = tf.layers.max_pooling1d(net, 2, 2, name="maxpool_6")
    net = _inception_1d(x=net, depth=4, name="Inception_7_1")
    net = _inception_1d(x=net, depth=4, name="Inception_7_2")
    net = tf.layers.max_pooling1d(net, 2, 2, name="maxpool_7")

    net = tf.layers.flatten(net, name="flat_layer")
    net = _norm_function(net, name="bn_dense_1")
    net = tf.layers.dense(net, 128, activation=tf.nn.relu,
                          bias_initializer=b_initializer, kernel_initializer=w_initializer,
                          kernel_regularizer=regularizer, name="dense_1")
    net = _norm_function(net, name="bn_dense_2")
    logits = tf.layers.dense(net, conf.num_cl, activation=None,
                             bias_initializer=b_initializer, kernel_initializer=w_initializer,
                             kernel_regularizer=regularizer, name="logits")
    return logits

