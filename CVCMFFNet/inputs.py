import tensorflow as tf


def placeholder_inputs(batch_size):

    images1_real = tf.placeholder(tf.float32, [batch_size, 256, 256, 3])
    images1_imag = tf.placeholder(tf.float32, [batch_size, 256, 256, 3])
    images2_real = tf.placeholder(tf.float32, [batch_size, 256, 256, 3])
    images2_imag = tf.placeholder(tf.float32, [batch_size, 256, 256, 3])
    ang = tf.placeholder(tf.float32, [batch_size, 256, 256, 3])

    labels = tf.placeholder(tf.int64, [batch_size, 256, 256])
    is_training = tf.placeholder(tf. bool)

    return images1_real, images1_imag, images2_real, images2_imag, ang, labels, is_training
