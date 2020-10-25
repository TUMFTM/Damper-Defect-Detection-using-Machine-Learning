import tensorflow as tf


def damnet_v1(x, keep_prob, is_train, conf, use_bn=True):

    def activation_1d(inputs, name):
        return tf.nn.tanh(inputs, name=name)

    def activation_2d(inputs, name):
        return tf.nn.elu(inputs, name=name)

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

    def _norm_function(input, name, train_phase):
        """wrapper function for batch norm layer"""
        return tf.layers.batch_normalization(inputs=input,
                                             training=train_phase,
                                             momentum=0.9,
                                             name=name,
                                             gamma_regularizer=r_ini)

    def _conv1d_bn_relu_layer(inputs, filters, kernel_size, nameext, padding='same', use_activation=True, use_bn=True):
        """wrapper function for convolution with batch norm (toggle bias) and relu non-linearity"""
        if use_bn:
            layer = tf.layers.conv1d(inputs=inputs,
                                     filters=filters,
                                     kernel_size=kernel_size,
                                     padding=padding,
                                     kernel_initializer=w_ini,
                                     bias_initializer=b_ini,
                                     kernel_regularizer=r_ini,
                                     use_bias=False,
                                     name='conv_' + nameext)
        else:
            layer = tf.layers.conv1d(inputs=inputs,
                                     filters=filters,
                                     kernel_size=kernel_size,
                                     padding=padding,
                                     kernel_initializer=w_ini,
                                     bias_initializer=b_ini,
                                     kernel_regularizer=r_ini,
                                     use_bias=True,
                                     name='conv_' + nameext)
        if use_bn:
            layer = _norm_function(layer, 'norm_conv_' + nameext, is_train)
        if use_activation:
            layer = activation_1d(layer, 'activation_' + nameext)
        return layer

    def _conv2d_bn_relu_layer(inputs, filters, kernel_size, nameext, padding='same', use_activation=True, use_bn=True):
        """wrapper function for convolution with batch norm (toggle bias) and relu non-linearity"""
        if use_bn:
            layer = tf.layers.conv2d(inputs=inputs,
                                     filters=filters,
                                     kernel_size=kernel_size,
                                     padding=padding,
                                     kernel_initializer=w_ini,
                                     bias_initializer=b_ini,
                                     kernel_regularizer=r_ini,
                                     use_bias=False,
                                     name='conv_' + nameext)
        else:
            layer = tf.layers.conv2d(inputs=inputs,
                                     filters=filters,
                                     kernel_size=kernel_size,
                                     padding=padding,
                                     kernel_initializer=w_ini,
                                     bias_initializer=b_ini,
                                     kernel_regularizer=r_ini,
                                     use_bias=True,
                                     name='conv_' + nameext)
        if use_bn:
            layer = _norm_function(layer, 'norm_conv_' + nameext, is_train)
        if use_activation:
            layer = activation_2d(layer, 'activation_' + nameext)
        return layer

    def new_fc_layer(input, num_outputs, train_phase, name, use_activation=True, use_bn=True):  # Use Batch Normalization
        with tf.variable_scope(name):
            w_ini, b_ini, reg_ini = initializers()
            if use_bn:  # omit bias
                layer = tf.layers.dense(inputs=input,
                                        units=num_outputs,
                                        kernel_initializer=w_ini,
                                        bias_initializer=b_ini,
                                        kernel_regularizer=reg_ini,
                                        use_bias=False,
                                        name='FC',
                                        )
            else:
                layer = tf.layers.dense(inputs=input,
                                        units=num_outputs,
                                        kernel_initializer=w_ini,
                                        bias_initializer=b_ini,
                                        kernel_regularizer=reg_ini,
                                        use_bias=True,
                                        name='FC',
                                        )
            if use_bn:
                layer = _norm_function(layer, 'BN', train_phase)
                tf.summary.histogram('BN', layer)
            if use_activation:
               layer = activation_1d(layer, 'activation') if conf.input == '1D' else activation_2d(layer,'activation')
            return layer

    def _inception_1d(x, depth, name, train_phase, use_bn=True):
        """
        Inception 1D module implementation.
        :param x: input to the current module (4D tensor with channels-last)
        :param depth: linearly controls the depth of the network (int)
        :param name: name of the variable scope (str)
        """
        with tf.variable_scope(name):
            x_norm = _norm_function(x, 'norm_input', train_phase) if use_bn else x
            # Branch 1:  conv 1x1
            branch_conv_3_3 = _conv1d_bn_relu_layer(inputs=x_norm, filters=16*depth, kernel_size=3, nameext='1_1', use_bn=use_bn)

            # Branch 2:  conv 3x3
            branch_conv_3_3_2 = _conv1d_bn_relu_layer(inputs=x_norm, filters=16, kernel_size=1, nameext='3_3_1', use_bn=use_bn)
            branch_conv_3_3_2 = _conv1d_bn_relu_layer(inputs=branch_conv_3_3_2, filters=16*depth, kernel_size=3, nameext='3_3_2', use_bn=use_bn)

            # Branch 3:  conv 5x5
            branch_conv_5_5 = _conv1d_bn_relu_layer(inputs=x_norm, filters=16, kernel_size=1, nameext='5_5_1', use_bn=use_bn)
            branch_conv_5_5 = _conv1d_bn_relu_layer(inputs=branch_conv_5_5, filters=16*depth, kernel_size=5, nameext='5_5_2', use_bn=use_bn)

            # Branch 4:  conv 7x7
            branch_conv_7_7 = _conv1d_bn_relu_layer(inputs=x_norm, filters=16, kernel_size=1, nameext='7_7_1', use_bn=use_bn)
            branch_conv_7_7 = _conv1d_bn_relu_layer(inputs=branch_conv_7_7, filters=16*depth, kernel_size=7, nameext='7_7_2', use_bn=use_bn)

            # Branch 5: (max_pool 3x3 + conv 1x1)
            branch_maxpool_3_3 = tf.layers.max_pooling1d(inputs=x_norm, pool_size=3, strides=1, padding="same", name="maxpool_3")
            if use_bn:
                branch_maxpool_3_3 = _norm_function(branch_maxpool_3_3, name="norm_maxpool_3_3", train_phase=train_phase)
                branch_maxpool_3_3 = tf.layers.conv1d(inputs=branch_maxpool_3_3, filters=16*depth, kernel_size=1, use_bias=False,
                                                      bias_initializer=b_ini, kernel_initializer=w_ini, activation=tf.nn.tanh,
                                                      kernel_regularizer=r_ini,
                                                      padding="same", name="conv_maxpool_3")
            else:
                branch_maxpool_3_3 = tf.layers.conv1d(inputs=branch_maxpool_3_3, filters=16*depth, kernel_size=1, use_bias=True,
                                                      bias_initializer=b_ini, kernel_initializer=w_ini, activation=tf.nn.tanh,
                                                      kernel_regularizer=r_ini,
                                                      padding="same", name="conv_maxpool_3")

            branch_sep_conv = tf.layers.separable_conv1d(x_norm, 16*depth, 7, padding='SAME', activation=tf.nn.tanh,
                                               depth_multiplier=16,
                                               depthwise_initializer=w_ini, pointwise_initializer=w_ini,
                                               bias_initializer=b_ini,
                                               depthwise_regularizer=r_ini, pointwise_regularizer=r_ini)
            # Concatenate
            output = tf.concat([branch_conv_3_3, branch_conv_3_3_2, branch_conv_5_5, branch_conv_7_7, branch_maxpool_3_3,
                                branch_sep_conv],
                               axis=-1)
            return output

    def _inception_2d(x, depth, name, train_phase, use_bn=True):
        """
        Inception 2D module implementation.
        :param x: input to the current module (4D tensor with channels-last)
        :param depth: linearly controls the depth of the network (int)
        :param name: name of the variable scope (str)
        """
        with tf.variable_scope(name):
            x_norm = _norm_function(x, 'norm_input', train_phase) if use_bn else x
            # Branch 1:  conv 3x3 (single)
            branch_conv_3_3 = _conv2d_bn_relu_layer(inputs=x_norm, filters=16 * depth, kernel_size=3, nameext='3_3',
                                                    use_bn=use_bn)

            # Branch 2:  conv 3x3
            branch_conv_3_3_2 = _conv2d_bn_relu_layer(inputs=x_norm, filters=16, kernel_size=1, nameext='3_3_1',
                                                      use_bn=use_bn)
            branch_conv_3_3_2 = _conv2d_bn_relu_layer(inputs=branch_conv_3_3_2, filters=16 * depth, kernel_size=3,
                                                      nameext='3_3_2', use_bn=use_bn)

            # Branch 3:  conv 5x5
            branch_conv_5_5 = _conv2d_bn_relu_layer(inputs=x_norm, filters=16, kernel_size=1, nameext='5_5_1',
                                                    use_bn=use_bn)
            branch_conv_5_5 = _conv2d_bn_relu_layer(inputs=branch_conv_5_5, filters=16 * depth, kernel_size=5,
                                                    nameext='5_5_2', use_bn=use_bn)

            # Branch 4:  conv 7x7
            branch_conv_7_7 = _conv2d_bn_relu_layer(inputs=x_norm, filters=16, kernel_size=1, nameext='7_7_1',
                                                    use_bn=use_bn)
            branch_conv_7_7 = _conv2d_bn_relu_layer(inputs=branch_conv_7_7, filters=16 * depth, kernel_size=7,
                                                    nameext='7_7_2', use_bn=use_bn)

            # Branch 5: (max_pool 3x3 + conv 1x1)
            branch_maxpool_3_3 = tf.layers.max_pooling2d(inputs=x_norm, pool_size=3, strides=1, padding="same",
                                                         name="maxpool_3")
            if use_bn:
                branch_maxpool_3_3 = _norm_function(branch_maxpool_3_3, name="norm_maxpool_3_3", train_phase=train_phase)
                branch_maxpool_3_3 = tf.layers.conv2d(inputs=branch_maxpool_3_3, filters=16*depth, kernel_size=1,
                                                      use_bias=False, activation=tf.nn.elu,
                                                      bias_initializer=b_ini, kernel_initializer=w_ini,
                                                      kernel_regularizer=r_ini,
                                                      padding="same", name="conv_maxpool_3")
            else:
                branch_maxpool_3_3 = tf.layers.conv2d(inputs=branch_maxpool_3_3, filters=16*depth, kernel_size=1,
                                                      use_bias=True, activation=tf.nn.elu,
                                                      bias_initializer=b_ini, kernel_initializer=w_ini,
                                                      kernel_regularizer=r_ini,
                                                      padding="same", name="conv_maxpool_3")

            branch_sep_conv = tf.layers.separable_conv2d(x_norm, 16*depth, 7, padding='SAME', activation=tf.nn.elu,
                                               depth_multiplier=16,
                                               depthwise_initializer=w_ini, pointwise_initializer=w_ini,
                                               bias_initializer=b_ini,
                                               depthwise_regularizer=r_ini, pointwise_regularizer=r_ini)
            # Concatenate
            output = tf.concat(
                [branch_conv_3_3, branch_conv_3_3_2, branch_conv_5_5, branch_conv_7_7, branch_maxpool_3_3, branch_sep_conv],
                axis=-1)
            return output

    def _global_average_pooling_1d(x):
        return tf.reduce_mean(x, axis=1, name='Global_avg_pooling')

    def _global_average_pooling_2d(x):
        return tf.reduce_mean(x, axis=[1, 2], name='Global_avg_pooling')

    def _squeeze_excitation_layer1d(input_x, out_dim, layer_name, ratio=16):
        with tf.variable_scope(layer_name):
            squeeze = _global_average_pooling_1d(input_x)
            excitation = tf.layers.dense(squeeze, units=out_dim // ratio, activation=tf.nn.relu,
                                         name=layer_name + '_fc1')
            excitation = tf.layers.dense(excitation, units=out_dim, activation=tf.nn.sigmoid,
                                         name=layer_name + '_fc2')
            excitation = tf.reshape(excitation, [-1, 1, out_dim])
            scale = input_x * excitation
            return scale

    def _squeeze_excitation_layer2d(input_x, out_dim, layer_name, ratio=16):
        with tf.variable_scope(layer_name):
            squeeze = _global_average_pooling_2d(input_x)
            excitation = tf.layers.dense(squeeze, units=out_dim // ratio, activation=tf.nn.relu,
                                         name=layer_name + '_fc')
            excitation = tf.layers.dense(excitation, units=out_dim, activation=tf.nn.sigmoid,
                                         name=layer_name + '_fc2')
            excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
            scale = input_x * excitation
            return scale

    filter_multiplier = 2
    if conf.input == '1D':
        x_multichannel1d = tf.reshape(x, [-1, conf.seq_lngth, conf.num_ch])
        net1d = _inception_1d(x_multichannel1d, filter_multiplier, 'DamCeption', is_train, use_bn)
        net1d = _squeeze_excitation_layer1d(net1d, net1d.shape[-1], 'SE', ratio=8)
        net1d = tf.layers.max_pooling1d(net1d, 2, 2)
        net1d = _conv1d_bn_relu_layer(net1d, 64, 3, '3', use_bn=use_bn, padding='valid')
        net1d = tf.layers.max_pooling1d(net1d, 2, 2)
        net1d = tf.layers.flatten(net1d, 'flat_layer')
        net1d = new_fc_layer(net1d, 64, is_train, "fc1_relu_bn", use_bn=use_bn)
        # apply dropout to first FC layer
        net1d = tf.nn.dropout(net1d, keep_prob, name='Dropout')
        # 2nd fully connected layer = output
        logits = tf.layers.dense(inputs=net1d, units=conf.num_cl, activation=None, kernel_initializer=w_ini,
                                 bias_initializer=b_ini, kernel_regularizer=r_ini, name='Output')
    else:
        x_multichannel2d = tf.reshape(x, [-1, conf.img_dim[0], conf.img_dim[1], conf.num_ch])
        net2d = _inception_2d(x_multichannel2d, filter_multiplier, 'DamCeption', is_train, use_bn)
        net2d = _squeeze_excitation_layer2d(net2d, net2d.shape[-1], 'SE', ratio=8)
        net2d = tf.layers.max_pooling2d(net2d, 2, 2)
        net2d = _conv2d_bn_relu_layer(net2d, 64, 3, '3', use_bn=use_bn, padding='valid')
        net2d = tf.layers.max_pooling2d(net2d, 2, 2)
        net2d = tf.layers.flatten(net2d, 'flat_layer')
        net2d = new_fc_layer(net2d, 64, is_train, "fc1_relu_bn", use_bn=use_bn)
        # apply dropout to first FC layer
        net2d = tf.nn.dropout(net2d, keep_prob, name='Dropout')
        # 2nd fully connected layer = output
        logits = tf.layers.dense(inputs=net2d, units=conf.num_cl, activation=None, kernel_initializer=w_ini,
                                 bias_initializer=b_ini, kernel_regularizer=r_ini, name='Output')
    return logits

