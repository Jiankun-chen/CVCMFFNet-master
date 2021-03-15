import os
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim
import CVCMFFNet
import scipy.io as scio
import time
import glob
import numpy as np
from PIL import Image
import math
from time import sleep
from PIL import ImageEnhance
from tqdm import tqdm
from tqdm import trange
import warnings

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

WORKING_DIR = os.getcwd()
TRAINING_DIR = os.path.join(WORKING_DIR, 'Data', 'Training')
TEST_DIR = os.path.join(WORKING_DIR, 'Data', 'Test1')
ROOT_LOG_DIR = os.path.join(WORKING_DIR, 'Output')
RUN_NAME = "model"
LOG_DIR = os.path.join(ROOT_LOG_DIR, RUN_NAME)
TRAIN_WRITER_DIR = os.path.join(LOG_DIR, 'Training')
TEST_WRITER_DIR = os.path.join(LOG_DIR, 'Test1')
CHECKPOINT_FN = 'model.ckpt'
CHECKPOINT_FL = os.path.join(LOG_DIR, CHECKPOINT_FN)

BATCH_NORM_DECAY = 0.95
MAX_STEPS = 30000
BATCH_SIZE = 8
SAVE_INTERVAL = 200


def main():
    training_data = CVCMFFNet.GetData(TRAINING_DIR)
    test_data = CVCMFFNet.GetData(TEST_DIR)
    g = tf.Graph()

    with g.as_default():

        images1_real, images1_imag, images2_real, images2_imag, ang, labels, is_training = CVCMFFNet.placeholder_inputs(batch_size=BATCH_SIZE)

        arg_scope = CVCMFFNet.inference_scope(is_training=True, batch_norm_decay=BATCH_NORM_DECAY)

        with slim.arg_scope(arg_scope):

            #logits, freal_layer1, fimag_layer1, freal_layer2, fimag_layer2, freal_layer3, fimag_layer3, freal_layer4, fimag_layer4, freal_layer5, fimag_layer5, freal_conv_1x1, fimag_conv_1x1, freal_conv_3x3_1, fimag_conv_3x3_1, freal_conv_3x3_2, fimag_conv_3x3_2, freal_conv_3x3_3, fimag_conv_3x3_3, freal_mix, fimag_mix, fang_2, fang_4, fang_6, fang_8, fang_out = SegNet.inference(images1_real, images1_imag, images2_real, images2_imag, ang, class_inc_bg=3)

            logits = CVCMFFNet.inference(images1_real, images1_imag, images2_real, images2_imag, ang, class_inc_bg=3)

        CVCMFFNet.add_output_images(images_real=images1_real, images_imag=images1_imag, logits=logits, labels=labels)

        loss = CVCMFFNet.loss_calc(logits=logits, labels=labels)

        train_op, global_step = CVCMFFNet.training(loss=loss, learning_rate=1e-05)

        total_parameters = CVCMFFNet.count()

        flops = CVCMFFNet.count_flops(g)

        accuracy, out_label = CVCMFFNet.evaluation(logits=logits, labels=labels)

        iou_class0, iou_class1, iou_class2, pa_0, pa_1, pa_2 = CVCMFFNet.iou_and_pa(logits=logits, labels=labels)

        summary = tf.summary.merge_all()

        init = tf.global_variables_initializer()

        saver = tf.train.Saver([x for x in tf.global_variables() if 'Adam' not in x.name])

        sm = tf.train.SessionManager()

        with sm.prepare_session("", init_op=init, saver=saver, checkpoint_dir=LOG_DIR) as sess:

            sess.run(tf.variables_initializer([x for x in tf.global_variables() if 'Adam' in x.name]))

            train_writer = tf.summary.FileWriter(TRAIN_WRITER_DIR, sess.graph)

            test_writer = tf.summary.FileWriter(TEST_WRITER_DIR)

            global_step_value, = sess.run([global_step])

            print("CVCMFF Net is Ready, Last Trained Iteration was:", global_step_value)

            print("Train Start...")

            sleep(0.5)

            ID = 0

            for step in range(global_step_value + 1, global_step_value + MAX_STEPS + 1, SAVE_INTERVAL):

                with trange(SAVE_INTERVAL, ncols=120, ascii='=>', mininterval=0.05) as t:

                    for sub_step in t:

                        images1_real_batch, images1_imag_batch, images2_real_batch, images2_imag_batch, ang_batch, labels_batch = training_data.next_batch(BATCH_SIZE)

                        train_feed_dict = {images1_real: images1_real_batch,
                                           images1_imag: images1_imag_batch,
                                           images2_real: images2_real_batch,
                                           images2_imag: images2_imag_batch,
                                           ang: ang_batch,
                                           labels: labels_batch,
                                           is_training: True}

                        #_, train_loss_value, train_accuracy_value,  freal_layer1, fimag_layer1, freal_layer2, fimag_layer2, freal_layer3, fimag_layer3, freal_layer4, fimag_layer4, freal_layer5, fimag_layer5, freal_conv_1x1, fimag_conv_1x1, freal_conv_3x3_1, fimag_conv_3x3_1, freal_conv_3x3_2, fimag_conv_3x3_2, freal_conv_3x3_3, fimag_conv_3x3_3, freal_mix, fimag_mix, fang_2, fang_4, fang_6, fang_8, fang_out, train_summary_str = sess.run([train_op, loss, accuracy, freal_layer1, fimag_layer1, freal_layer2, fimag_layer2, freal_layer3, fimag_layer3, freal_layer4, fimag_layer4, freal_layer5, fimag_layer5, freal_conv_1x1, fimag_conv_1x1, freal_conv_3x3_1, fimag_conv_3x3_1, freal_conv_3x3_2, fimag_conv_3x3_2, freal_conv_3x3_3, fimag_conv_3x3_3, freal_mix, fimag_mix, fang_2, fang_4, fang_6, fang_8, fang_out, summary], feed_dict=train_feed_dict)
                        _, train_loss_value, train_accuracy_value, train_iou_class0, train_iou_class1, train_iou_class2, train_summary_str = sess.run([train_op, loss, accuracy, iou_class0, iou_class1, iou_class2, summary], feed_dict=train_feed_dict)
                        train_mean_iou = (train_iou_class0 + train_iou_class1 + train_iou_class2) / 3
                        step_now = step + sub_step
                        t.set_description('Training')
                        t.set_postfix(Iteration=step_now, Loss=train_loss_value, Accuarcy=train_accuracy_value, MIoU=train_mean_iou)

                sleep(0.5)

                train_writer.add_summary(train_summary_str, step)
                train_writer.flush()

                images1_real_batch, images1_imag_batch, images2_real_batch, images2_imag_batch, ang_batch, labels_batch = test_data.next_batch(BATCH_SIZE)

                test_feed_dict = {images1_real: images1_real_batch,
                                  images1_imag: images1_imag_batch,
                                  images2_real: images2_real_batch,
                                  images2_imag: images2_imag_batch,
                                  ang: ang_batch,
                                  labels: labels_batch,
                                  is_training: False}

                test_loss_value, test_accuracy_value, test_iou_class0, test_iou_class1, test_iou_class2, test_pa_0, test_pa_1, test_pa_2, test_summary_str, test_label, img_real, img_imag = sess.run([loss, accuracy, iou_class0, iou_class1, iou_class2, pa_0, pa_1, pa_2, summary, out_label, images1_real, images1_imag], feed_dict=test_feed_dict)

                test_mean_iou = (test_iou_class0 + test_iou_class1 + test_iou_class2) / 3

                test_mpa = (test_pa_1 + test_pa_1 + test_pa_2) / 3

                '''
                # Feature visualization

                #scio.savemat("images1_real.mat", {"images1_real": images1_real})
                #scio.savemat("images1_imag.mat", {"images1_imag": images1_imag})
                #scio.savemat("images2_real.mat", {"images2_real": images2_real})
                #scio.savemat("images2_imag.mat", {"images2_imag": images2_imag})
                #scio.savemat("ang.mat", {"ang": ang})
                scio.savemat("freal_layer1.mat", {"freal_layer1": freal_layer1})
                scio.savemat("fimag_layer1.mat", {"fimag_layer1": fimag_layer1})
                scio.savemat("freal_layer2.mat", {"freal_layer2": freal_layer2})
                scio.savemat("fimag_layer2.mat", {"fimag_layer2": fimag_layer2})
                scio.savemat("freal_layer3.mat", {"freal_layer3": freal_layer3})
                scio.savemat("fimag_layer3.mat", {"fimag_layer3": fimag_layer3})
                scio.savemat("freal_layer4.mat", {"freal_layer4": freal_layer4})
                scio.savemat("fimag_layer4.mat", {"fimag_layer4": fimag_layer4})
                scio.savemat("freal_layer5.mat", {"freal_layer5": freal_layer5})
                scio.savemat("fimag_layer5.mat", {"fimag_layer5": fimag_layer5})
                scio.savemat("freal_conv_1x1.mat", {"freal_conv_1x1": freal_conv_1x1})
                scio.savemat("fimag_conv_1x1.mat", {"fimag_conv_1x1": fimag_conv_1x1})
                scio.savemat("freal_conv_3x3_1.mat", {"freal_conv_3x3_1": freal_conv_3x3_1})
                scio.savemat("fimag_conv_3x3_1.mat", {"fimag_conv_3x3_1": fimag_conv_3x3_1})
                scio.savemat("freal_conv_3x3_2.mat", {"freal_conv_3x3_2": freal_conv_3x3_2})
                scio.savemat("fimag_conv_3x3_2.mat", {"fimag_conv_3x3_2": fimag_conv_3x3_2})
                scio.savemat("freal_conv_3x3_3.mat", {"freal_conv_3x3_3": freal_conv_3x3_3})
                scio.savemat("fimag_conv_3x3_3.mat", {"fimag_conv_3x3_3": fimag_conv_3x3_3})
                scio.savemat("freal_mix.mat", {"freal_mix": freal_mix})
                scio.savemat("fimag_mix.mat", {"fimag_mix": fimag_mix})
                scio.savemat("fang_2.mat", {"fang_2": fang_2})
                scio.savemat("fang_4.mat", {"fang_4": fang_4})
                scio.savemat("fang_6.mat", {"fang_6": fang_6})
                scio.savemat("fang_8.mat", {"fang_8": fang_8})
                scio.savemat("fang_out.mat", {"fang_out": fang_out})
                '''

                print("Test Loss:", test_loss_value, "Test accuracy:", test_accuracy_value)
                print("Test IoU_class0:", test_iou_class0, "Test IoU_class1:", test_iou_class1, "Test IoU_class2:", test_iou_class2, "Test MIoU:", test_mean_iou)
                print("Test Pix_accuary0:", test_pa_0, "Test Pix_accuary1:", test_pa_1, "Test Pix_accuary2:", test_pa_2, "Test MPA:", test_mpa)

                scio.savemat("{ID}".format(ID=ID) + ".mat", {"label": test_label})

                test_writer.add_summary(test_summary_str, step)
                test_writer.flush()

                # testlabel2outtag
                data_path = "{ID}".format(ID=ID) + ".mat"
                output_path = "./outtag/"
                data = scio.loadmat(data_path)
                label = np.array(data['label'])
                R = np.zeros([256, 256, 2])
                G = np.zeros([256, 256, 2])
                B = np.zeros([256, 256, 2])
                for k in range(2):
                    for i in range(256):
                        for j in range(256):
                            if (label[i, j, k] == 0):
                                [R[i, j, k], G[i, j, k], B[i, j, k]] = [0, 0, 0]
                            if (label[i, j, k] == 1):
                                [R[i, j, k], G[i, j, k], B[i, j, k]] = [128, 0, 0]
                            if (label[i, j, k] == 2):
                                [R[i, j, k], G[i, j, k], B[i, j, k]] = [0, 128, 0]
                # save outtag as png
                for k in range(2):
                    R1 = Image.fromarray(R[:, :, k]).convert('L')
                    G1 = Image.fromarray(G[:, :, k]).convert('L')
                    B1 = Image.fromarray(B[:, :, k]).convert('L')
                    image = Image.merge("RGB", (R1, G1, B1))
                    image.save(output_path + str(ID + k) + ".png", 'png')
                print("Outtag is OK")
                ID = ID + 2

                # delete tag.mat
                for root, dirs, files in os.walk(WORKING_DIR):
                    for name in files:
                        if name.endswith(".mat"):
                            os.remove(os.path.join(root, name))
                            #print("Delete File: " + os.path.join(root, name))

                saver.save(sess, CHECKPOINT_FL, global_step=step)
                print("CHECKPOINT Saved")
                print("================")


if __name__ == '__main__':
    main()
