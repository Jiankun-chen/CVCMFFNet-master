import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from .layers import unpool_with_argmax
from tensorflow.contrib import layers as layers_lib


def inference_scope(is_training, batch_norm_decay=0.95):
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        normalizer_fn=slim.batch_norm,
                        stride=1,
                        padding='SAME'):

        with slim.arg_scope([slim.batch_norm],
                            is_training=is_training,
                            decay=batch_norm_decay) as sc:
            return sc


def cv_conv(I, Q, input_channel, n, name=None):  # 这里用到卷积核共享， n是卷积核数量
    with tf.variable_scope(name):
        filter1 = tf.Variable(tf.truncated_normal(shape=[3, 3, input_channel, n], stddev=0.01))
        filter2 = tf.Variable(tf.truncated_normal(shape=[3, 3, input_channel, n], stddev=0.01))

        # II = slim.conv2d(I, n, [3, 3])
        II = tf.nn.conv2d(I, filter1, strides=[1, 1, 1, 1], padding='SAME')
        # IQ = slim.conv2d(I, n, [3, 3])
        IQ = tf.nn.conv2d(I, filter2, strides=[1, 1, 1, 1], padding='SAME')
        # QI = slim.conv2d(Q, n, [3, 3])
        QI = tf.nn.conv2d(Q, filter1, strides=[1, 1, 1, 1], padding='SAME')
        # QQ = slim.conv2d(Q, n, [3, 3])
        QQ = tf.nn.conv2d(Q, filter2, strides=[1, 1, 1, 1], padding='SAME')
        real = II - QQ
        imag = IQ + QI
        axis_real = list(range(len(real.get_shape()) - 1))
        mean_real, variance_real = tf.nn.moments(real, axis_real)
        real = tf.nn.batch_normalization(real, mean_real, variance_real, offset=0, scale=1, variance_epsilon=1e-5)
        axis_imag = list(range(len(imag.get_shape()) - 1))
        mean_imag, variance_imag = tf.nn.moments(imag, axis_imag)
        imag = tf.nn.batch_normalization(imag, mean_imag, variance_imag, offset=0, scale=1, variance_epsilon=1e-5)
        real = tf.nn.relu(real)
        imag = tf.nn.relu(imag)
        return real, imag


def cv_rate_conv(I, Q, n, size, stride, rate):  # 这里用到卷积核共享， n是卷积核数量

    # II = slim.conv2d(I, n, [3, 3])
    II = layers_lib.conv2d(I, n, size, stride=stride, rate=rate)
    # IQ = slim.conv2d(I, n, [3, 3])
    IQ = layers_lib.conv2d(I, n, size, stride=stride, rate=rate)
    # QI = slim.conv2d(Q, n, [3, 3])
    QI = layers_lib.conv2d(Q, n, size, stride=stride, rate=rate)
    # QQ = slim.conv2d(Q, n, [3, 3])
    QQ = layers_lib.conv2d(Q, n, size, stride=stride, rate=rate)
    real = II - QQ
    imag = IQ + QI
    axis_real = list(range(len(real.get_shape()) - 1))
    mean_real, variance_real = tf.nn.moments(real, axis_real)
    real = tf.nn.batch_normalization(real, mean_real, variance_real, offset=0, scale=1, variance_epsilon=1e-5)
    axis_imag = list(range(len(imag.get_shape()) - 1))
    mean_imag, variance_imag = tf.nn.moments(imag, axis_imag)
    imag = tf.nn.batch_normalization(imag, mean_imag, variance_imag, offset=0, scale=1, variance_epsilon=1e-5)
    real = tf.nn.relu(real)
    imag = tf.nn.relu(imag)
    return real, imag


def inference(images1_real, images1_imag, images2_real, images2_imag, ang, class_inc_bg=None):
    # print(images)
    # c = tf.transpose(images, [0, 2, 1, 3])           #这里将WH调转回来
    # shape = c.shape
    # c = images.transpose([0, 2, 1, 3])
    '''for m in range(5):
        d = c[m, :, :, 0]
        d_type = type(d)
        print(d_type)
        g_real = []
        g_imag = []'''
    # real = tf.real(images)
    # imag = tf.imag(images)
    real1 = images1_real
    imag1 = images1_imag
    real2 = images2_real
    imag2 = images2_imag

    '''
    for i in range(shape[1]):
            for j in range(shape[2]):
                e = d[i, j]                           #这里双层嵌套循环
                f_real = tf.real(e)                         #提取实部
                f_imag = tf.imag(e)                         #提取虚部
                g_real.append(f_real)                 #由实数部分生成的g_real，为list
                g_imag.append(f_imag)                 #由实数部分生成的g_real，为list
        s_real = np.array(g_real, dtype=float)
        s_imag = np.array(g_imag, dtype=float)
        r_real = s_real.reshape(512, 512)
        r_imag = s_imag.reshape(512, 512)

        real = np.tile(r_real, (3, 1, 1))            #扩充为三通道
        imag = np.tile(r_imag, (3, 1, 1))
        '''
    mol1 = tf.sqrt(tf.square(images1_real) + tf.square(images1_imag))
    tf.summary.image('input', mol1, max_outputs=20)

    with tf.variable_scope('pool1'):
        # net = slim.conv2d(images, 64, [3, 3], scope='conv1_1')
        real1_1, imag1_1 = cv_conv(real1, imag1, 3, 64, name='conv1_1')
        sub_real1_1, sub_imag1_1 = cv_conv(real2, imag2, 3, 64, name='sub_conv1_1')

        # net = slim.conv2d(net, 64, [3, 3], scope='conv1_2')
        real1_2, imag1_2 = cv_conv(real1_1, imag1_1, 64, 32, name='conv1_2')
        # freal_layer1 = real1_2
        # fimag_layer1 = imag1_2
        sub_real1_2, sub_imag1_2 = cv_conv(sub_real1_1, sub_imag1_1, 64, 32, name='sub_conv1_2')

        mix_real1 = tf.concat([real1_2, sub_real1_2], axis=3, name='mix_real1')
        mix_imag1 = tf.concat([imag1_2, sub_imag1_2], axis=3, name='mix_imag1')

        net = tf.sqrt(mix_real1**2 + mix_imag1**2)

        # sub_net = tf.sqrt(sub_real1_2**2 + sub_imag1_2**2)
        net, arg1 = tf.nn.max_pool_with_argmax(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool1')
        # sub_net, sub_arg1 = tf.nn.max_pool_with_argmax(sub_net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='sub_maxpool1')
        real1 = tf.nn.avg_pool(mix_real1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='avgpool_re1')
        sub_real1 = tf.nn.avg_pool(sub_real1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='sub_avgpool_re1')
        imag1 = tf.nn.avg_pool(mix_imag1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='avgpool_im1')
        sub_imag1 = tf.nn.avg_pool(sub_imag1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='sub_avgpool_im1')

    with tf.variable_scope('pool2'):
        # net = slim.conv2d(net, 128, [3, 3], scope='conv2_1')
        real2_1, imag2_1 = cv_conv(real1, imag1, 64, 128, name='conv2_1')
        sub_real2_1, sub_imag2_1 = cv_conv(sub_real1, sub_imag1, 32, 128, name='sub_conv2_1')

        # net = slim.conv2d(net, 128, [3, 3], scope='conv2_2')
        real2_2, imag2_2 = cv_conv(real2_1, imag2_1, 128, 64, name='conv2_2')
        # freal_layer2 = real2_2
        # fimag_layer2 = imag2_2

        sub_real2_2, sub_imag2_2 = cv_conv(sub_real2_1, sub_imag2_1, 128, 64, name='sub_conv2_2')

        mix_real2 = tf.concat([real2_2, sub_real2_2], axis=3, name='mix_real2')
        mix_imag2 = tf.concat([imag2_2, sub_imag2_2], axis=3, name='mix_imag2')

        net = tf.sqrt(mix_real2**2 + mix_imag2**2)
        # sub_net = tf.sqrt(sub_real2_2**2 + sub_imag2_2**2)
        net, arg2 = tf.nn.max_pool_with_argmax(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool2')
        # sub_net, sub_arg2 = tf.nn.max_pool_with_argmax(sub_net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='sub_maxpool2')

        real2 = tf.nn.avg_pool(mix_real2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='avgpool_re2')
        sub_real2 = tf.nn.avg_pool(sub_real2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='sub_avgpool_re2')
        imag2 = tf.nn.avg_pool(mix_imag2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='avgpool_im2')
        sub_imag2 = tf.nn.avg_pool(sub_imag2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='sub_avgpool_im2')

    with tf.variable_scope('pool3'):
        # net = slim.conv2d(net, 256, [3, 3], scope='conv3_1')
        real3_1, imag3_1 = cv_conv(real2, imag2, 128, 256, name='conv3_1')
        sub_real3_1, sub_imag3_1 = cv_conv(sub_real2, sub_imag2, 64, 256, name='sub_conv3_1')

        # net = slim.conv2d(net, 256, [3, 3], scope='conv3_2')
        real3_2, imag3_2 = cv_conv(real3_1, imag3_1, 256, 256, name='conv3_2')
        sub_real3_2, sub_imag3_2 = cv_conv(sub_real3_1, sub_imag3_1, 256, 256, name='sub_conv3_2')

        # net = slim.conv2d(net, 256, [3, 3], scope='conv3_3')
        real3_3, imag3_3 = cv_conv(real3_2, imag3_2, 256, 128, name='conv3_3')
        # freal_layer3 = real3_3
        # fimag_layer3 = imag3_3

        sub_real3_3, sub_imag3_3 = cv_conv(sub_real3_2, sub_imag3_2, 256, 128, name='sub_conv3_3')

        mix_real3 = tf.concat([real3_3, sub_real3_3], axis=3, name='mix_real3')
        mix_imag3 = tf.concat([imag3_3, sub_imag3_3], axis=3, name='mix_imag3')

        net = tf.sqrt(mix_real3**2 + mix_imag3**2)
        # sub_net = tf.sqrt(sub_real3_3**2 + sub_imag3_3**2)
        net, arg3 = tf.nn.max_pool_with_argmax(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool3')
        # sub_net, sub_arg3 = tf.nn.max_pool_with_argmax(sub_net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='sub_maxpool3')

        real3 = tf.nn.avg_pool(mix_real3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='avgpool_re3')
        sub_real3 = tf.nn.avg_pool(sub_real3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='sub_avgpool_re3')
        imag3 = tf.nn.avg_pool(mix_imag3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='avgpool_im3')
        sub_imag3 = tf.nn.avg_pool(sub_imag3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='sub_avgpool_im3')

    with tf.variable_scope('pool4'):
        # net = slim.conv2d(net, 512, [3, 3], scope='conv4_1')
        real4_1, imag4_1 = cv_conv(real3, imag3, 256, 512, name='conv4_1')
        sub_real4_1, sub_imag4_1 = cv_conv(sub_real3, sub_imag3, 128, 512, name='sub_conv4_1')

        # net = slim.conv2d(net, 512, [3, 3], scope='conv4_2')
        real4_2, imag4_2 = cv_conv(real4_1, imag4_1, 512, 512, name='conv4_2')
        sub_real4_2, sub_imag4_2 = cv_conv(sub_real4_1, sub_imag4_1, 512, 512, name='sub_conv4_2')

        # net = slim.conv2d(net, 512, [3, 3], scope='conv4_3')
        real4_3, imag4_3 = cv_conv(real4_2, imag4_2, 512, 256, name='conv4_3')
        # freal_layer4 = real4_3
        # fimag_layer4 = imag4_3

        sub_real4_3, sub_imag4_3 = cv_conv(sub_real4_2, sub_imag4_2, 512, 256, name='sub_conv4_3')

        mix_real4 = tf.concat([real4_3, sub_real4_3], axis=3, name='mix_real4')
        mix_imag4 = tf.concat([imag4_3, sub_imag4_3], axis=3, name='mix_imag4')

        net = tf.sqrt(mix_real4**2 + mix_imag4**2)
        # sub_net = tf.sqrt(sub_real4_3**2 + sub_imag4_3**2)

        net, arg4 = tf.nn.max_pool_with_argmax(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool4')
        # sub_net, sub_arg4 = tf.nn.max_pool_with_argmax(sub_net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='sub_maxpool4')

        real4 = tf.nn.avg_pool(mix_real4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='avgpool_re4')
        sub_real4 = tf.nn.avg_pool(sub_real4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='sub_avgpool_re4')
        imag4 = tf.nn.avg_pool(mix_imag4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='avgpool_im4')
        sub_imag4 = tf.nn.avg_pool(sub_imag4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='sub_avgpool_im4')

    with tf.variable_scope('pool5'):
        # net = slim.conv2d(net, 512, [3, 3], scope='conv5_1')
        real5_1, imag5_1 = cv_conv(real4, imag4, 512, 512, name='conv5_1')
        sub_real5_1, sub_imag5_1 = cv_conv(sub_real4, sub_imag4, 256, 512, name='sub_conv5_1')

        # net = slim.conv2d(net, 512, [3, 3], scope='conv5_2')
        real5_2, imag5_2 = cv_conv(real5_1, imag5_1, 512, 512, name='conv5_2')
        sub_real5_2, sub_imag5_2 = cv_conv(sub_real5_1, sub_imag5_1, 512, 512, name='sub_conv5_2')

        # net = slim.conv2d(net, 512, [3, 3], scope='conv5_3')
        real5_3, imag5_3 = cv_conv(real5_2, imag5_2, 512, 256, name='conv5_3')
        # freal_layer5 = real5_3
        # fimag_layer5 = imag5_3

        sub_real5_3, sub_imag5_3 = cv_conv(sub_real5_2, sub_imag5_2, 512, 256, name='sub_conv5_3')

        mix_real5 = tf.concat([real5_3, sub_real5_3], axis=3, name='mix_real5')
        mix_imag5 = tf.concat([imag5_3, sub_imag5_3], axis=3, name='mix_imag5')

        net = tf.sqrt(mix_real5**2 + mix_imag5**2)
        # sub_net = tf.sqrt(sub_real5_3**2 + sub_imag5_3**2)

        net, arg5 = tf.nn.max_pool_with_argmax(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool5')
        # sub_net, sub_arg5 = tf.nn.max_pool_with_argmax(sub_net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='sub_maxpool5')

        real5 = tf.nn.avg_pool(mix_real5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='avgpool_re5')
        # sub_real5 = tf.nn.avg_pool(sub_real5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='sub_avgpool_re5')
        imag5 = tf.nn.avg_pool(mix_imag5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='avgpool_im5')
        # sub_imag5 = tf.nn.avg_pool(sub_imag5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='sub_avgpool_im5')
        # mix_real5 = tf.concat([real5, sub_real5], axis=3, name='mix_real5')
        # mix_imag5 = tf.concat([imag5, sub_imag5], axis=3, name='mix_imag5')

        # The atrous spatial pyramid pooling
        conv_1x1_real, conv_1x1_imag = cv_rate_conv(real5, imag5, 128, [1, 1], 1, 1)
        # freal_conv_1x1 = conv_1x1_real
        # fimag_conv_1x1 = conv_1x1_imag

        conv_3x3_1_real, conv_3x3_1_imag = cv_rate_conv(real5, imag5, 128, [3, 3], stride=1, rate=2)
        # freal_conv_3x3_1 = conv_3x3_1_real
        # fimag_conv_3x3_1 = conv_3x3_1_imag

        conv_3x3_2_real, conv_3x3_2_imag = cv_rate_conv(real5, imag5, 128, [3, 3], stride=1, rate=4)
        # freal_conv_3x3_2 = conv_3x3_2_real
        # fimag_conv_3x3_2 = conv_3x3_2_imag

        conv_3x3_3_real, conv_3x3_3_imag = cv_rate_conv(real5, imag5, 128, [3, 3], stride=1, rate=6)
        # freal_conv_3x3_3 = conv_3x3_3_real
        # fimag_conv_3x3_3 = conv_3x3_3_imag

        image_level_features_real = tf.reduce_mean(real5, [1, 2], name='global_average_pooling', keepdims=True)
        image_level_features_real = layers_lib.conv2d(image_level_features_real, 512, [1, 1], stride=1, scope='conv_1x1_2_real')
        image_level_features_real = tf.image.resize_bilinear(image_level_features_real, tf.shape(real5)[1:3], name='upsample_real')
        image_level_features_imag = tf.reduce_mean(imag5, [1, 2], name='global_average_pooling', keepdims=True)
        image_level_features_imag = layers_lib.conv2d(image_level_features_imag, 512, [1, 1], stride=1, scope='conv_1x1_2_imag')
        image_level_features_imag = tf.image.resize_bilinear(image_level_features_imag, tf.shape(real5)[1:3], name='upsample_imag')

        net_assp_real = tf.concat([conv_1x1_real, conv_3x3_1_real, conv_3x3_2_real, conv_3x3_3_real, image_level_features_real], axis=3, name='concat_real')
        net_assp_imag = tf.concat([conv_1x1_imag, conv_3x3_1_imag, conv_3x3_2_imag, conv_3x3_3_imag, image_level_features_imag], axis=3, name='concat_imag')

        real5_4 = layers_lib.conv2d(net_assp_real, 512, [1, 1], stride=1, scope='conv_1x1_real_concat')
        imag5_4 = layers_lib.conv2d(net_assp_imag, 512, [1, 1], stride=1, scope='conv_1x1_imag_concat')
        # freal_mix = real5_4
        # fimag_mix = imag5_4

    with tf.variable_scope('unpool5'):
        real5u = unpool_with_argmax(real5_4, arg5, name='maxunpool_real5')
        imag5u = unpool_with_argmax(imag5_4, arg5, name='maxunpool_imag5')

        # net = slim.conv2d(net, 512, [3, 3], scope='uconv5_3')
        real5_3u, imag5_3u = cv_conv(real5u, imag5u, 512, 512, name='uconv5_3')
        # net = slim.conv2d(net, 512, [3, 3], scope='uconv5_2')
        real5_2u, imag5_2u = cv_conv(real5_3u, imag5_3u, 512, 512, name='uconv5_2')
        # net = slim.conv2d(net, 512, [3, 3], scope='uconv5_1')
        real5_1u, imag5_1u = cv_conv(real5_2u, imag5_2u, 512, 512, name='uconv5_1')

    with tf.variable_scope('unpool4'):
        real4u = unpool_with_argmax(real5_1u, arg4, name='maxunpool_real4')
        imag4u = unpool_with_argmax(imag5_1u, arg4, name='maxunpool_imag4')

        # net = slim.conv2d(net, 512, [3, 3], scope='uconv4_3')
        real4_3u, imag4_3u = cv_conv(real4u, imag4u, 512, 512, name='uconv4_3')
        # net = slim.conv2d(net, 512, [3, 3], scope='uconv4_2')
        real4_2u, imag4_2u = cv_conv(real4_3u, imag4_3u, 512, 512, name='uconv4_2')
        # net = slim.conv2d(net, 256, [3, 3], scope='uconv4_1')
        real4_1u, imag4_1u = cv_conv(real4_2u, imag4_2u, 512, 256, name='uconv4_1')

    with tf.variable_scope('unpool3'):
        real3u = unpool_with_argmax(real4_1u, arg3, name='maxunpool_real3')
        imag3u = unpool_with_argmax(imag4_1u, arg3, name='maxunpool_imag3')

        # net = slim.conv2d(net, 256, [3, 3], scope='uconv3_3')
        real3_3u, imag3_3u = cv_conv(real3u, imag3u, 256, 256, name='uconv3_3')
        # net = slim.conv2d(net, 256, [3, 3], scope='uconv3_2')
        real3_2u, imag3_2u = cv_conv(real3_3u, imag3_3u, 256, 256, name='uconv3_2')
        # net = slim.conv2d(net, 128, [3, 3], scope='uconv3_1')
        real3_1u, imag3_1u = cv_conv(real3_2u, imag3_2u, 256, 128, name='uconv3_1')

    with tf.variable_scope('unpool2'):
        real2u = unpool_with_argmax(real3_1u, arg2, name='maxunpool_real2')
        imag2u = unpool_with_argmax(imag3_1u, arg2, name='maxunpool_imag2')

        # net = slim.conv2d(net, 128, [3, 3], scope='uconv2_2')
        real2_2u, imag2_2u = cv_conv(real2u, imag2u, 128, 128, name='uconv2_2')
        # net = slim.conv2d(net, 64, [3, 3], scope='uconv2_1')
        real2_1u, imag2_1u = cv_conv(real2_2u, imag2_2u, 128, 64, name='uconv2_1')

    with tf.variable_scope('unpool1'):
        real1u = unpool_with_argmax(real2_1u, arg1, name='maxunpool_real1')
        imag1u = unpool_with_argmax(imag2_1u, arg1, name='maxunpool_imag1')

        # net = slim.conv2d(net, 64, [3, 3], scope='uconv1_2')
        real1_1u, imag1_1u = cv_conv(real1u, imag1u, 64, 64, name='uconv1_2')
        net = tf.sqrt(real1_1u**2 + imag1_1u**2)

    with tf.variable_scope('ang'):
        ang_0 = slim.conv2d(ang, 32, [3, 3], scope='ang_0')
        ang_1 = slim.conv2d(ang_0, 32, [3, 3], scope='ang_1')
        ang_2 = slim.conv2d(ang_1, 32, [3, 3], scope='ang_2')
        # fang_2 = ang_2

        ang_1_cat = tf.concat([ang_0, ang_2], axis=3, name='ang_1_cat')

        ang_3 = slim.conv2d(ang_1_cat, 64, [3, 3], scope='ang_3')
        ang_4 = slim.conv2d(ang_3, 64, [3, 3], scope='ang_4')
        # fang_4 = ang_4

        ang_2_cat = tf.concat([ang_1_cat, ang_4], axis=3, name='ang_2_cat')

        ang_5 = slim.conv2d(ang_2_cat, 128, [3, 3], scope='ang_5')
        ang_6 = slim.conv2d(ang_5, 128, [3, 3], scope='ang_6')
        # fang_6 = ang_6

        ang_3_cat = tf.concat([ang_2_cat, ang_6], axis=3, name='ang_3_cat')

        ang_7 = slim.conv2d(ang_3_cat, 256, [3, 3], scope='ang_7')
        ang_8 = slim.conv2d(ang_7, 256, [3, 3], scope='ang_8')
        # fang_8 = ang_8

        ang_4_cat = tf.concat([ang_3_cat, ang_8], axis=3, name='ang_4_cat')

        ang_out = slim.conv2d(ang_4_cat, 64, [3, 3], scope='ang_out')
        # fang_out = ang_out

    net_inf = tf.concat([net, ang_out], axis=3, name='net_inf')
    logits = slim.conv2d(net_inf, class_inc_bg, [3, 3], scope='uconv1_1')

    # return logits,  freal_layer1, fimag_layer1, freal_layer2, fimag_layer2, freal_layer3, fimag_layer3, freal_layer4, fimag_layer4, freal_layer5, fimag_layer5, freal_conv_1x1, fimag_conv_1x1, freal_conv_3x3_1, fimag_conv_3x3_1, freal_conv_3x3_2, fimag_conv_3x3_2, freal_conv_3x3_3, fimag_conv_3x3_3, freal_mix, fimag_mix, fang_2, fang_4, fang_6, fang_8, fang_out
    return logits
