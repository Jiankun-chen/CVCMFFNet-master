import tensorflow as tf


def add_output_images(images_real, images_imag, logits, labels):
    images = tf.sqrt(tf.square(images_real) + tf.square(images_imag))
    cast_labels = tf.cast(labels, tf.uint8)
    cast_labels = tf.cast(cast_labels[..., None], tf.float32) * 128
    tf.summary.image('input_labels', cast_labels, max_outputs=20)

    classification0 = tf.nn.softmax(logits=logits, dim=-1)[..., 0:1]
    # classification0 = tf.squeeze(input=classification0, squeeze_dims=[4], name='classification0')
    classification1 = tf.nn.softmax(logits=logits, dim=-1)[..., 1:2]
    # classification1 = tf.squeeze(input=classification1, squeeze_dims=[4], name='classification1')
    # classification1 = tf.nn.softmax(logits=logits, dim=-1)[...,1]
    classification2 = tf.nn.softmax(logits=logits, dim=-1)[..., 2:3]
    # classification2 = tf.squeeze(input=classification2, squeeze_dims=[4], name='classification2')
    # output_image_gb = images[..., 0]
    output_image_r = (1 - classification2) + tf.multiply(images[..., 0:1], classification2)
    output_image_g = (1 - classification1) + tf.multiply(images[..., 0:1], classification1)
    output_image_b = (1 - classification0) + tf.multiply(tf.cast(128, tf.float32), classification0)
    output_image = tf.stack([0.5 * (output_image_r + output_image_g), 0.5 * output_image_g, 0.5 * (output_image_b + output_image_g)], axis=3)
    output_image = tf.squeeze(output_image)
    tf.summary.image('output_mixed', output_image, max_outputs=20)

    output_image_binary = tf.argmax(logits, 3)
    output_image_binary = tf.cast(output_image_binary[..., None], tf.float32) * 128
    tf.summary.image('output_labels', output_image_binary, max_outputs=20)

    # dcrf_label = tf.cast(dcrf_label[..., None], tf.float32) * 128
    # tf.summary.image('dcrf_label', dcrf_label, max_outputs=20)

    # 由于制作dcrf_mix，需要遍历每一个像素点并使用判断语句，因此不适合在训练中加入
    # 又由于close_open函数不适用于tensor，因此也不能在训练中加入。
    return


'''
    output_image_binary = tf.argmax(logits, 3)
    output_image_binary = tf.cast(output_image_binary[...,None], tf.float32) * 128/255
    tf.summary.image('output_labels', output_image_binary, max_outputs=20)

    output_labels_mixed_r = output_image_binary[...,0] + tf.multiply(images[...,0], (1-output_image_binary[...,0]))
    output_labels_mixed = tf.stack([output_labels_mixed_r, output_image_gb, output_image_gb], axis=3)
    tf.summary.image('output_labels_mixed', output_labels_mixed, max_outputs=20)
'''
