import tensorflow as tf
from models.SqueezeNet import squeezenet1d
from models.LeNet import lenet5_1d
from models.WavCeption import wavception
from models.SE_ResNeXt import se_resnext_1d, _squeeze_excitation_layer1d
from models.DamNet import damnet_v1
from models.Baseline_new import baseline_new
from models.Paper import paper_nets
from models.VGGNet import vgg_11_1d
from models.MLP import mlp
import numpy as np


def conv_net(x, keep_prob, train_phase, conf):
    # split the signal into unknown number of samples with given length for every channel
    x_multichannel = tf.reshape(x, [-1, conf.seq_lngth, conf.num_ch])

    def initializers(bias_init=conf.bias_init, beta=conf.l2_str):
        weight_init = tf.initializers.variance_scaling(scale=2.0,       # He initialization https://arxiv.org/pdf/1502.01852v1.pdf
                                                       mode='fan_in',
                                                       distribution='normal'
                                                       )
        bias_ini = tf.constant_initializer(value=bias_init)                  # small constant
        regularizer = tf.contrib.layers.l2_regularizer(scale=beta)
        return weight_init, bias_ini, regularizer

    def new_conv_layer(input,  # The previous layer.
                       num_filters,  # Number of filters.
                       filter_size,  # Width of each filter.
                       train_phase,
                       name,
                       stride=1,
                       padding="SAME",
                       use_relu=True,
                       use_bn=True):
        with tf.variable_scope(name):
            w_ini, b_ini, reg_ini = initializers()
            if use_bn:  # omit bias
                layer = tf.layers.conv1d(inputs=input,
                                         filters=num_filters,
                                         kernel_size=filter_size,
                                         padding=padding,
                                         strides=stride,
                                         kernel_initializer=w_ini,
                                         bias_initializer=b_ini,
                                         kernel_regularizer=reg_ini,
                                         use_bias=False,
                                         name='conv1D')
            else:
                layer = tf.layers.conv1d(inputs=input,
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
                tf.summary.histogram('activity',layer)

            # BN after non-linearity performs better
            if use_bn:
                layer = tf.layers.batch_normalization(inputs=layer,
                                                      training=train_phase,
                                                      name='BN',
                                                      gamma_regularizer=reg_ini)
            return layer

    def new_fc_layer(input,  # The previous layer.
                     num_outputs,  # Num. outputs.
                     train_phase,
                     name,  # Name
                     use_bn=True):  # Use Batch Normalization
        with tf.variable_scope(name):
            w_ini, b_ini, reg_ini = initializers()
            if use_bn:  # omit bias
                layer = tf.layers.dense(inputs=input,
                                        units=num_outputs,
                                        kernel_initializer=w_ini,
                                        bias_initializer=b_ini,
                                        kernel_regularizer=reg_ini,
                                        use_bias=False,
                                        activation=tf.nn.relu,
                                        name='FC',
                                        )
            else:
                layer = tf.layers.dense(inputs=input,
                                        units=num_outputs,
                                        kernel_initializer=w_ini,
                                        bias_initializer=b_ini,
                                        kernel_regularizer=reg_ini,
                                        use_bias=True,
                                        activation=tf.nn.relu,
                                        name='FC',
                                        )
            if use_bn:
                layer = tf.layers.batch_normalization(layer,
                                                      training=train_phase,
                                                      name='BN',
                                                      gamma_regularizer=reg_ini)
                tf.summary.histogram('BN', layer)
            return layer

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
                split_layer = tf.layers.conv1d(x_split[i], filters, filter_size, padding='SAME', activation=tf.nn.relu,
                                               kernel_initializer=w_ini, kernel_regularizer=reg_ini, bias_initializer=b_ini)
                layers_split.append(split_layer)
                filter_sum += filters
            layer = tf.concat(layers_split, axis=-1)
            layer = tf.layers.conv1d(layer, num_filters, 1, activation=tf.nn.relu, padding='SAME', bias_initializer=b_ini,
                                     kernel_initializer=w_ini, kernel_regularizer=reg_ini,name='conv_1x1')
        return layer

    def inference_baseline(x_multichannel, keep_prob, is_train, num_classes):
        w_ini, b_ini, reg_ini = initializers()
        use_bn = False
        # first block
        net = new_conv_layer(x_multichannel, conf.k, conf.extent, is_train, 'conv1_relu_bn', use_bn=use_bn)
        # net = _squeeze_excitation_layer1d(net, 16, 'se_block')
        net = tf.layers.max_pooling1d(net, 2, 2, name='maxpool1')
        # second block
        # net = new_conv_layer(net, 32, 3, is_train, 'conv2_relu_bn', use_bn=use_bn)
        # net = tf.layers.max_pooling1d(net, 2, 2, name='maxpool1')

        net = tf.layers.flatten(net, 'flat_layer')
        net = new_fc_layer(net, conf.fc_n, is_train, "fc1_relu_bn", use_bn=use_bn)
        # apply dropout to first FC layer
        net = tf.nn.dropout(net, keep_prob, name='Dropout')
        # 2nd fully connected layer = output
        logits = tf.layers.dense(inputs=net, units=num_classes, activation=None, kernel_initializer=w_ini,
                                 bias_initializer=b_ini, kernel_regularizer=reg_ini, name='Output')
        return logits

    def inference_doubleconv(x_multichannel, keep_prob, is_train, num_classes):
        # Architecture see Verstraete_2017
        w_ini, b_ini, reg_ini = initializers()
        use_bn = False

        with tf.variable_scope("Block1"):
            net = new_conv_layer(x_multichannel, 32, 3, is_train, name="conv1_relu_bn", use_bn=use_bn)
            net = new_conv_layer(net, 32, 3, is_train, name='conv2_relu_bn', use_bn=use_bn)
            net = tf.layers.max_pooling1d(net, 2, 2, name="maxpool1")
        with tf.variable_scope("Block2"):
            net = new_conv_layer(net, 64, 3, is_train, name="conv1_relu_bn", use_bn=use_bn)
            net = new_conv_layer(net, 64, 3, is_train, name='conv2_relu_bn', use_bn=use_bn)
            net = tf.layers.max_pooling1d(net, 2, 2, name="maxpool2")
        with tf.variable_scope("Block3"):
            net = new_conv_layer(net, 128, 3, is_train, name="conv1_relu_bn", use_bn=use_bn)
            net = new_conv_layer(net, 128, 3, is_train, name='conv2_relu_bn', use_bn=use_bn)
            net = tf.layers.max_pooling1d(net, 2, 2, name="maxpool3")
        with tf.variable_scope("FC-Block"):
            net = tf.layers.flatten(net,name="flat_layer")
            net = new_fc_layer(net, 100, is_train, name="fc1_w_relu", use_bn=use_bn)
            net = tf.nn.dropout(net, keep_prob,name="fc1_drop")
            net = new_fc_layer(net, 100, is_train, name="fc2_w_relu", use_bn=use_bn)
            net = tf.nn.dropout(net, keep_prob, name="fc2_drop")
            logits = tf.layers.dense(inputs=net,
                                     units=num_classes,
                                     activation=None,
                                     kernel_initializer=w_ini,
                                     bias_initializer=b_ini,
                                     kernel_regularizer=reg_ini,
                                     name='Output')
        return logits

    def ticnn(x_multichannel, keep_proba, is_train, num_classes):
        # Zhang, Wei; Li, Chuanhao; Peng, Gaoliang; Chen, Yuanhang; Zhang, Zhujun
        # A deep convolutional neural network with new training methods for
        # bearing fault diagnosis under noisy environment and different working load
        # DOI 10.1016/j.ymssp.2017.06.022
        # see section 4.2 and table 2 (paper) for architectural details
        net = new_conv_layer(name="Convolution1",       #apply "same" padding, use ReLU activation
                             input=x_multichannel,
                             filter_size=4,    # THP_MOD: original paper uses seq_length of 2048, filter_size=64, stride=8
                             num_filters=16,
                             train_phase=is_train,
                             stride=1)  # paper applies BN before activation, this layer implements it after non-linearity
        net = tf.nn.dropout(net, keep_proba, name="Kernel_dropout")
        net = tf.layers.max_pooling1d(net, 2, 2, name="Pooling1")
        net = new_conv_layer(net, 32, 3, is_train, "Convolution2")  # "same" padding, stride of 1, use bn, use relu
        net = tf.layers.max_pooling1d(net, 2, 2, name="Pooling2")
        net = new_conv_layer(net, 64, 3, is_train, "Convolution3")  # "same" padding, stride of 1, use bn, use relu
        net = tf.layers.max_pooling1d(net, 2, 2, name="Pooling3")
        # net = new_conv_layer(net, 64, 3, is_train, "Convolution4")  # "same" padding, stride of 1, use bn, use relu
        # net = tf.layers.max_pooling1d(net, 2, 2, name="Pooling4")
        # net = new_conv_layer(net, 64, 3, is_train, "Convolution5")  # "same" padding, stride of 1, use bn, use relu
        # net = tf.layers.max_pooling1d(net, 2, 2, name="Pooling5")
        net = new_conv_layer(net, 64, 3, is_train, "Convolution6", padding="VALID")  # no padding, stride of 1, use bn, use relu
        net = tf.layers.max_pooling1d(net, 2, 2, name="Pooling6")
        net = tf.layers.flatten(net,"flat_layer")
        net = new_fc_layer(net, 100, is_train, "Fully-connected")
        # output layer (softmax applied at loss function)
        logits = tf.layers.dense(net, num_classes, activation=None, name='logits')
        return logits

    def sepconv(x_multichannel, keep_proba, is_train, num_classes):
        w_ini, b_ini, reg_ini = initializers()
        net = tf.layers.separable_conv1d(x_multichannel, 64, 9, padding='SAME', activation=tf.nn.relu,
                                         depth_multiplier=32,
                                         depthwise_initializer=w_ini, pointwise_initializer=w_ini,
                                         bias_initializer=b_ini,
                                         depthwise_regularizer=reg_ini, pointwise_regularizer=reg_ini)
        # net = _squeeze_excitation_layer1d(net, 64, 'SE_layer', ratio=8)
        net = tf.layers.max_pooling1d(net, 2, 2, name='maxpool1')

        net = tf.layers.flatten(net, 'flat_layer')
        net = new_fc_layer(net, 128, is_train, "fc1_relu_bn", use_bn=False)
        # apply dropout to first FC layer
        net = tf.nn.dropout(net, keep_proba, name='Dropout')
        # 2nd fully connected layer = output
        logits = tf.layers.dense(inputs=net, units=num_classes, activation=None, kernel_initializer=w_ini,
                                 bias_initializer=b_ini, kernel_regularizer=reg_ini, name='Output')
        return logits

    def grouped_sepconv(x_multichannel, keep_proba, is_train, num_classes, splits):
        w_ini, b_ini, reg_ini = initializers()
        # splits = [1,1,1,1,1,1]  # specify the grouping of channels (currently they have to be next to each other)

        net = grouped_sepconv_layer(x_multichannel, splits, 64, 9)
        net = tf.layers.max_pooling1d(net, 2, 2, name='maxpool1')

        net = tf.layers.flatten(net, 'flat_layer')
        net = new_fc_layer(net, 128, is_train, "fc1_relu_bn", use_bn=False)
        # apply dropout to first FC layer
        net = tf.nn.dropout(net, keep_proba, name='Dropout')
        # 2nd fully connected layer = output
        logits = tf.layers.dense(inputs=net, units=num_classes, activation=None, kernel_initializer=w_ini,
                                 bias_initializer=b_ini, kernel_regularizer=reg_ini, name='Output')
        return logits


    if conf.model == 'baseline':
        out = inference_baseline(x_multichannel, keep_prob=keep_prob, is_train=train_phase, num_classes=conf.num_cl)
    elif conf.model == 'baseline_new':
        out = baseline_new(x, keep_prob, train_phase, conf)
    elif conf.model == 'mlp':
        out = mlp(x, keep_prob, train_phase, conf)
    elif conf.model == 'doubleconv':
        out = inference_doubleconv(x_multichannel, keep_prob=keep_prob, is_train=train_phase, num_classes=conf.num_cl)
    elif conf.model == 'SqueezeNet':
        out = squeezenet1d(x, keep_prob=keep_prob, is_train=train_phase, conf=conf)
    elif conf.model == 'LeNet':
        out = lenet5_1d(x, keep_prob=keep_prob, is_train=train_phase, conf=conf)
    elif conf.model == 'TICNN':
        out = ticnn(x_multichannel, keep_prob, train_phase, conf.num_cl)
        # set mini batch size to 8, note that original paper uses batch size 10, ensemble of nets for prediction and changing dropout rate
    elif conf.model == 'WavCeption':
        out = wavception(x, keep_prob, train_phase, conf)
        # no dropout in wavception model, might want to add
    elif conf.model == 'SE_ResNeXt':
        out = se_resnext_1d(x, train_phase, conf)
    elif conf.model == 'DamNet_v1':
        out = damnet_v1(x, keep_prob, train_phase, conf, use_bn=False)
    elif conf.model == 'sepConv':
        out = sepconv(x_multichannel, keep_prob, train_phase, conf.num_cl)
    elif conf.model == 'grouped_sepConv':
        out = grouped_sepconv(x_multichannel, keep_prob, train_phase, conf.num_cl, [2, 2, 3])
    elif conf.model == 'VGG':
        out = vgg_11_1d(x, keep_prob, conf)
    elif conf.model == 'paper_prep':
        out = paper_nets(x, keep_prob, conf)
    else:
        raise Warning('model parameter "%s" ill defined' % conf.model)
    return out
