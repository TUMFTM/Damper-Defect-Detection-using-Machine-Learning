import tensorflow as tf


def paper_nets(x, keep_prob, conf):

    def activation(inputs, name):
        return tf.nn.relu(inputs, name=name)

    def initializers(bias_init=conf.bias_init, l2str=conf.l2_str):
        weight_init = tf.initializers.variance_scaling(scale=2.0,
                                                       # He initialization https://arxiv.org/pdf/1502.01852v1.pdf
                                                       mode='fan_in',
                                                       distribution='normal'
                                                       )
        bias_ini = tf.constant_initializer(value=bias_init)  # small constant
        regularizer = tf.contrib.layers.l2_regularizer(scale=l2str)
        return weight_init, bias_ini, regularizer
    w_ini, b_ini, r_ini = initializers()

    def _conv1d_relu_layer(inputs, filters, kernel_size, nameext, padding='same', use_activation=True):
        """wrapper function for convolution with batch norm (toggle bias) and relu non-linearity"""
        layer = tf.layers.conv1d(inputs=inputs,
                                 filters=filters,
                                 kernel_size=kernel_size,
                                 padding=padding,
                                 kernel_initializer=w_ini,
                                 bias_initializer=b_ini,
                                 kernel_regularizer=r_ini,
                                 use_bias=True,
                                 name='conv_' + nameext)
        if use_activation:
            layer = activation(layer, 'activation_' + nameext)
        return layer

    def _conv2d_relu_layer(inputs, filters, kernel_size, nameext, padding='same', use_activation=True):
        """wrapper function for convolution with batch norm (toggle bias) and relu non-linearity"""
        layer = tf.layers.conv2d(inputs=inputs,
                                 filters=filters,
                                 kernel_size=kernel_size,
                                 padding=padding,
                                 kernel_initializer=w_ini,
                                 bias_initializer=b_ini,
                                 kernel_regularizer=r_ini,
                                 use_bias=True,
                                 name='conv_' + nameext)
        if use_activation:
            layer = activation(layer, 'activation_' + nameext)
        return layer

    def _fc_layer(input, num_outputs, name, use_activation=True):
        with tf.variable_scope(name):
            w_ini, b_ini, reg_ini = initializers()
            layer = tf.layers.dense(inputs=input,
                                    units=num_outputs,
                                    kernel_initializer=w_ini,
                                    bias_initializer=b_ini,
                                    kernel_regularizer=reg_ini,
                                    use_bias=True,
                                    name='FC',
                                    )
            if use_activation:
                layer = activation(layer, 'activation')
            return layer

    def _inception_1d(x, name):
        """
        Inception 1D module implementation.
        :param x: input to the current module (4D tensor with channels-last)
        :param name: name of the variable scope (str)
        """
        with tf.variable_scope(name):
            # Branch 1:  conv 1x1
            branch_conv_1 = _conv1d_relu_layer(inputs=x, filters=16, kernel_size=1, nameext='1_1')

            # Branch 2:  conv 3x3
            branch_conv_3 = _conv1d_relu_layer(inputs=x, filters=16, kernel_size=3, nameext='3_3')

            # Branch 3:  conv 5x5
            branch_conv_5 = _conv1d_relu_layer(inputs=x, filters=16, kernel_size=5, nameext='5_5')

            # Branch 4:  conv 7x7
            branch_conv_7 = _conv1d_relu_layer(inputs=x, filters=16, kernel_size=7, nameext='7_7')

            # Branch 5:  conv 9x9
            branch_conv_9 = _conv1d_relu_layer(inputs=x, filters=16, kernel_size=9, nameext='9_9')

            # Branch 6:  conv 11x11
            branch_conv_11 = _conv1d_relu_layer(inputs=x, filters=16, kernel_size=11, nameext='11_11')

            # Concatenate
            output = tf.concat([branch_conv_1, branch_conv_3, branch_conv_5, branch_conv_7, branch_conv_9,
                                branch_conv_11],
                               axis=-1)
            return output

    def _inception_2d(x, name):
        """
        Inception 1D module implementation.
        :param x: input to the current module (4D tensor with channels-last)
        :param name: name of the variable scope (str)
        """
        with tf.variable_scope(name):
            # Branch 1:  conv 1x1
            branch_conv_1 = _conv2d_relu_layer(inputs=x, filters=16, kernel_size=1, nameext='1_1')

            # Branch 2:  conv 3x3
            branch_conv_3 = _conv2d_relu_layer(inputs=x, filters=16, kernel_size=3, nameext='3_3')

            # Branch 3:  conv 5x5
            branch_conv_5 = _conv2d_relu_layer(inputs=x, filters=16, kernel_size=5, nameext='5_5')

            # Branch 4:  conv 7x7
            branch_conv_7 = _conv2d_relu_layer(inputs=x, filters=16, kernel_size=7, nameext='7_7')

            # Branch 5:  conv 9x9
            branch_conv_9 = _conv2d_relu_layer(inputs=x, filters=16, kernel_size=9, nameext='9_9')

            # Branch 6:  conv 11x11
            branch_conv_11 = _conv2d_relu_layer(inputs=x, filters=16, kernel_size=11, nameext='11_11')

            # Concatenate
            output = tf.concat([branch_conv_1, branch_conv_3, branch_conv_5, branch_conv_7, branch_conv_9,
                                branch_conv_11],
                               axis=-1)
            return output
    if conf.input == '1D':
        # split signals according to length and channels
        x_multichannel = tf.reshape(x, [-1, conf.seq_lngth, conf.num_ch])
        # inference
        net = _inception_1d(x_multichannel, 'Inception')
        net = tf.layers.max_pooling1d(net, 2, 2, name='pool1')
        net = tf.layers.flatten(net, 'flat_layer')
        net = _fc_layer(net, 64, 'fully_conn')
        net = tf.nn.dropout(net, keep_prob, name='dropout')
        # output layer (softmax applied at loss function)
        logits = _fc_layer(net, conf.num_cl, name='logits', use_activation=False)
    else:
        # split signals according to length and channels
        x_multichannel = tf.reshape(x, [-1, conf.img_dim[0], conf.img_dim[1], conf.num_ch])
        # inference
        net = _inception_2d(x_multichannel, 'Inception')
        net = tf.layers.max_pooling2d(net, 2, 2, name='pool1')
        net = tf.layers.flatten(net, 'flat_layer')
        net = _fc_layer(net, 64, 'fully_conn')
        net = tf.nn.dropout(net, keep_prob, name='dropout')
        # output layer (softmax applied at loss function)
        logits = _fc_layer(net, conf.num_cl, name='logits', use_activation=False)
    return logits
