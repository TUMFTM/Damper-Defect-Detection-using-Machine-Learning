import tensorflow as tf


def baseline_new(x, keep_prob, is_train, conf):
    add_conv = conf.add_conv
    kernels = conf.kernels
    firstlayer_extent = conf.firstlayer_e
    pooling_size = conf.pooling_size
    layer_extent = 3

    def initializers(bias_init=conf.bias_init, beta=conf.l2_str):
        weight_init = tf.initializers.variance_scaling(scale=2.0,
                                                       # He initialization https://arxiv.org/pdf/1502.01852v1.pdf
                                                       mode='fan_in',
                                                       distribution='normal'
                                                       )
        bias_ini = tf.constant_initializer(value=bias_init)  # small constant
        regularizer = tf.contrib.layers.l2_regularizer(scale=beta)
        return weight_init, bias_ini, regularizer

    def new_conv1d_layer(inputs,  # The previous layer.
                         num_filters,  # Number of filters.
                         filter_size,  # Width of each filter.
                         name,
                         stride=1,
                         padding="SAME",
                         use_relu=True):
        with tf.variable_scope(name):
            w_ini, b_ini, reg_ini = initializers()
            layer = tf.layers.conv1d(inputs=inputs,
                                     filters=num_filters,
                                     kernel_size=filter_size,
                                     padding=padding,
                                     strides=stride,
                                     kernel_initializer=w_ini,
                                     bias_initializer=b_ini,
                                     kernel_regularizer=reg_ini,
                                     use_bias=True,
                                     name='conv1D')
            # Rectified Linear Unit (ReLU).
            if use_relu:
                layer = tf.nn.relu(layer, name='ReLU')
                tf.summary.histogram('activity', layer)
            return layer

    def new_conv2d_layer(inputs,  # The previous layer.
                         num_filters,  # Number of filters.
                         filter_size,  # Width of each filter.
                         name,
                         stride=1,
                         padding="SAME",
                         use_relu=True):
        with tf.variable_scope(name):
            w_ini, b_ini, reg_ini = initializers()
            layer = tf.layers.conv2d(inputs=inputs,
                                     filters=num_filters,
                                     kernel_size=filter_size,
                                     padding=padding,
                                     strides=stride,
                                     kernel_initializer=w_ini,
                                     bias_initializer=b_ini,
                                     kernel_regularizer=reg_ini,
                                     use_bias=True,
                                     name='conv2D')
            # Rectified Linear Unit (ReLU).
            if use_relu:
                layer = tf.nn.relu(layer, name='ReLU')
                tf.summary.histogram('activity', layer)
            return layer

    def new_fc_layer(inputs,  # The previous layer.
                     num_outputs,  # Num. outputs.
                     name):  # Name
        with tf.variable_scope(name):
            w_ini, b_ini, reg_ini = initializers()
            layer = tf.layers.dense(inputs=inputs,
                                    units=num_outputs,
                                    kernel_initializer=w_ini,
                                    bias_initializer=b_ini,
                                    kernel_regularizer=reg_ini,
                                    use_bias=True,
                                    activation=tf.nn.relu,
                                    name='FC',
                                    )
            return layer

    w_ini, b_ini, reg_ini = initializers()

    if conf.input == '1D':
        x_multichannel = tf.reshape(x, [-1, conf.seq_lngth, conf.num_ch])
        cntStage = 0
        with tf.variable_scope('Stage_'+str(cntStage)):
            net = new_conv1d_layer(x_multichannel, kernels, firstlayer_extent, 'conv_0')
            for j in range(add_conv):
                net = new_conv1d_layer(net, kernels, layer_extent, 'conv_' + str(j+1))
            net = tf.layers.max_pooling1d(net, pooling_size, pooling_size, name='maxpool_'+str(cntStage))

    elif conf.input == '2D':
        x_multichannel = tf.reshape(x, [-1, conf.img_dim[0], conf.img_dim[1], conf.num_ch])
        cntStage = 0
        with tf.variable_scope('Stage_' + str(cntStage)):
            net = new_conv2d_layer(x_multichannel, kernels, firstlayer_extent, 'conv_0')
            for j in range(add_conv):
                net = new_conv2d_layer(net, kernels, layer_extent, 'conv_' + str(j + 1))
            net = tf.layers.max_pooling2d(net, pooling_size, pooling_size, name='maxpool_' + str(cntStage))

    # original code
    net = tf.layers.flatten(net, 'flat_layer')
    net = new_fc_layer(net, 128, "fc1_relu_bn")

    # apply dropout to first FC layer
    net = tf.nn.dropout(net, keep_prob, name='Dropout')
    # 2nd fully connected layer = output
    logits = tf.layers.dense(inputs=net, units=conf.num_cl, activation=None, kernel_initializer=w_ini,
                             bias_initializer=b_ini, kernel_regularizer=reg_ini, name='Output')
    return logits
