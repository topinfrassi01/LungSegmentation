import tensorflow as tf
import numpy as np
import time
import os

from tensorflow.contrib.layers import batch_norm, softmax
from tensorflow.python.layers.convolutional import conv2d, conv2d_transpose
from tensorflow.python.layers.core import dropout
from tensorflow.python.layers.pooling import max_pooling2d

from metrics import hausdorff_object_score, dice_object_score, object_f_score
from utils import augment_batch

__author__ = "Mathias Baltzersen and Rasmus Hvingelby"

class SegmentModel:
    def __init__(self, hps):
        self.hps = hps

        self.batch_size = hps.get("batch_size")
        self.epochs = hps.get("epochs")

        self.num_classes = 3
        self.img_input_channels = 1

        self.threshold = tf.constant(0.5, dtype=tf.float32)
        self.l2_scale = hps.get("l2_scale")
        self.contour_loss_weight = hps.get("contour_loss_weight")


        self.dropout_p = 0.5
        self.learning_rate_value = hps.get("lr")

        self.sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=False,
            gpu_options={'allow_growth': True})
        )

        self._create_model()
        self.saver = tf.train.Saver()

        self.sess.run(tf.global_variables_initializer())

        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print("Total parameters of model.: {}".format(total_parameters))

    def train(self, input_image, gt_image, gt_cont):
        """
        Trains the model on given data epoch times

        :param input_image:
        :param gt_image:
        """

        summary_writer = tf.summary.FileWriter("./" + self.hps.get("exp_name"))

        for epoch in range(self.epochs):
            step = self._train_one_epoch(gt_cont, gt_image, input_image, summary_writer)
            
            if epoch % 10 == 0:
                self.saver.save(self.sess, self.hps.get('exp_name') + '/model', global_step=step)

        self.saver.save(self.sess, self.hps.get('exp_name') + '/final-model')

    def _get_image_descriptors(self, input_images):
        image_descriptors = []

        for batch in self._get_batches(input_images):
            feed_dict = {
                self.input_image: batch,
                self.dropout_prob: 1.0,
            }

            batch_image_descriptors = self.sess.run(self.img_descriptor, feed_dict=feed_dict)
            image_descriptors.extend(batch_image_descriptors)

        return np.array(image_descriptors)

    def _get_batches(self, input_images):
        num_examples = input_images.shape[0]
        num_batches = np.ceil(num_examples / self.batch_size)
        batches = np.array_split(input_images, num_batches)
        return batches

    def _get_dropout_predictions(self, input_images):

        predictions = []

        for batch in self._get_batches(input_images):
            feed_dict = {
                self.input_image: batch,
                self.dropout_prob: self.dropout_p
            }

            batch_pred = self.sess.run(self.preds, feed_dict=feed_dict)
            predictions.extend(batch_pred)

        return predictions


    def _train_one_epoch(self, gt_cont, gt_image, input_image, summary_writer):
        num_examples = input_image.shape[0]
        # shuffle
        permutation_idx = np.random.permutation(num_examples)

        shuffled_input_images = input_image[permutation_idx]
        shuffled_gt_images = gt_image[permutation_idx]
        shuffled_gt_conts = gt_cont[permutation_idx]

        # TODO : Changer pour ImageDataGenerator de Keras

        num_batches = num_examples / self.batch_size
        input_image_batches = np.array_split(shuffled_input_images, num_batches)
        gt_image_batches = np.array_split(shuffled_gt_images, num_batches)
        gt_cont_batches = np.array_split(shuffled_gt_conts, num_batches)

        for input_image_batch, gt_image_batch, gt_cont_batch in zip(input_image_batches, gt_image_batches,
                                                                    gt_cont_batches):

            x_train, y_train_seg, y_train_cont = augment_batch(input_image_batch, gt_image_batch, gt_cont_batch,
                                                               self.hps.get("img_size"))

            feed_dict = {
                self.input_image: x_train,
                self.gt_image: y_train_seg,
                self.gt_contours: y_train_cont,
                self.dropout_prob: self.dropout_p,
                self.lr: self.learning_rate_value
            }

            _, summary, step, loss = self.sess.run([self.train_op, self.summaries, self.global_step, self.loss],
                                                   feed_dict=feed_dict)

            print("step: {0:}, loss: {1:.3f}, training_data_size: {2:}".format(step, loss, num_examples))
            if step > 10000:
                self.learning_rate_value = 0.00005

            summary_writer.add_summary(summary, step)

        return step


    def _forward_pass(self, input_images):
        """
        Makes a forward pass without dropout to get
        predictions and img descriptors for. Used
        when bootstrap models need to predict.

        :param input_images:
        :return:
        """
        predictions = []
        image_descriptors = []

        for batch in self._get_batches(input_images):

            feed_dict = {
                self.input_image: batch,
                self.dropout_prob: 1.0
            }

            batch_pred, batch_img_descriptor = self.sess.run([self.preds, self.img_descriptor], feed_dict=feed_dict)

            predictions.extend(batch_pred)
            image_descriptors.extend(batch_img_descriptor)

        return predictions, image_descriptors

    def evaluate(self, input_images, gt_images, ensemble_count=1):

        num_examples = input_images.shape[0]
        num_batches = np.ceil(num_examples // self.batch_size)

        input_image_batches = np.array_split(input_images, num_batches)
        gt_image_batches = np.array_split(gt_images, num_batches)

        dropout_probability = 1.0

        ensembles = []

        if ensemble_count > 1:
            dropout_probability = 0.3

        for _ in range(ensemble_count):

            batch_predictions = []

            for i, (input_image_batch, gt_image_batch) in enumerate(zip(input_image_batches, gt_image_batches)):
                feed_dict = {
                    self.input_image: input_image_batch,
                    self.dropout_prob: dropout_probability
                }

                output_predictions = self.sess.run(self.preds, feed_dict=feed_dict)

                batch_predictions.extend(output_predictions)

            ensembles.append(np.array(batch_predictions))

        ensembles_predictions = np.array(ensembles)

        expected_shape_out = ensemble_count, input_images.shape[0], input_images.shape[1], input_images.shape[2], input_images.shape[3]
#        assert ensembles_predictions.shape == expected_shape_out

        return np.array(ensembles)

    def final_predictions(self, ensemble_predictions, soft=True):
        if soft:
            return np.mean(ensemble_predictions, axis=0)
        else:
            print("hard is not implemented")
            return np.mean(ensemble_predictions, axis=0)

    def _bottleneck(self, inputs, size=None):
        conv1 = conv2d(inputs, filters=size, kernel_size=1, padding='same')
        ac2 = self._add_common_layers(conv1)
        conv2 = conv2d(ac2, filters=size, kernel_size=3, padding='same')
        ac3 = self._add_common_layers(conv2)
        conv3 = conv2d(ac3, filters=size * 4, kernel_size=1, padding='same')

        # This 1x1 conv is used to match the dimension of x and F(x)
        hack_conv = conv2d(inputs, filters=size * 4, kernel_size=1, padding='same')

        return tf.add(hack_conv, conv3)

    def _add_common_layers(self, inputs):
        bn = batch_norm(inputs)
        relu_ = tf.nn.relu(bn)

        return relu_

    def _upsample(self, inputs, k):
        x = inputs
        for i in reversed(range(0, k)):
            x = conv2d_transpose(inputs=x, filters=self.num_classes * 2 ** i, kernel_size=4, strides=2, padding='same')
            x = dropout(x, rate=self.dropout_prob)
            x = self._add_common_layers(x)

        return x

    def _add_train_op(self):
        with tf.name_scope('loss'):
            #Log_loss is negative by tf definition
            seg_loss = tf.losses.log_loss(labels=self.gt_image, predictions=self.preds_seg)
            cont_loss = tf.losses.log_loss(labels=self.gt_contours, predictions=self.preds_cont,
                                           weights=self.contour_loss_weight)

            vars = tf.trainable_variables()

            #Apply regularization to all non bias variables
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars
                               if 'bias' not in v.name]) * self.l2_scale

            loss = tf.add_n([lossL2, seg_loss, cont_loss])

            self.loss = tf.reduce_mean(loss, name='loss')

        tf.summary.scalar('loss', self.loss)

        #TODO: In DCAN they use SGD.
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    def _create_model(self):

        self.input_image = tf.placeholder(tf.float32, shape=(None, None, None, self.img_input_channels), name='input_image_placeholder')
        self.gt_image = tf.placeholder(tf.int32, shape=(None, None, None, self.num_classes), name='gt_image_placeholder')
        self.gt_contours = tf.placeholder(tf.int32, shape=(None, None, None, self.num_classes), name='gt_contours_placeholder')
        self.dropout_prob = tf.placeholder(dtype=tf.float32, shape=None, name='dropout_prob_placeholder')

        self.lr = tf.placeholder(dtype=tf.float32, shape=None, name='learning_rate_placeholder')

        scale_nc = self.hps.get('scale_nc')

        with tf.variable_scope("encoder"):
            with tf.variable_scope("block_1"):
                conv1 = self._add_common_layers(conv2d(self.input_image, filters=32*scale_nc, kernel_size=3, padding='same'))
                conv2 = self._add_common_layers(conv2d(conv1, filters=32*scale_nc, kernel_size=3, padding='same'))

            with tf.variable_scope("block_2"):
                mp2 = max_pooling2d(conv2, pool_size=2, strides=2, padding='same')
                bn1 = self._add_common_layers(self._bottleneck(mp2, size=64*scale_nc))
                bn2 = self._add_common_layers(self._bottleneck(bn1, size=64*scale_nc))

            with tf.variable_scope("block_3"):
                mp3 = max_pooling2d(bn2, pool_size=2, strides=2, padding='same')
                bn3 = self._add_common_layers(self._bottleneck(mp3, size=128*scale_nc))
                bn4 = self._add_common_layers(self._bottleneck(bn3, size=128*scale_nc))

            with tf.variable_scope("block_4"):
                mp4 = max_pooling2d(bn4, pool_size=2, strides=2, padding='same')
                bn5 = self._add_common_layers(self._bottleneck(mp4, size=256*scale_nc))
                bn6 = self._add_common_layers(self._bottleneck(bn5, size=256*scale_nc))
                d1 = dropout(bn6, rate=self.dropout_prob)

            with tf.variable_scope("block_5"):
                mp5 = max_pooling2d(d1, pool_size=2, strides=2, padding='same')
                bn7 = self._add_common_layers(self._bottleneck(mp5, size=256*scale_nc))
                bn8 = self._add_common_layers(self._bottleneck(bn7, size=256*scale_nc))
                d2 = dropout(bn8, rate=self.dropout_prob)

            with tf.variable_scope("block_6"):
                mp6 = max_pooling2d(d2, pool_size=2, strides=2, padding='same')
                bn9 = self._add_common_layers(self._bottleneck(mp6, size=256*scale_nc))
                bn10 = self._add_common_layers(self._bottleneck(bn9, size=256*scale_nc))
                d3 = dropout(bn10, rate=self.dropout_prob)

        self.img_descriptor = tf.reduce_mean(d3, axis=(1, 2))

        with tf.variable_scope("decoder_seg"):
            deconvs = []
            deconvs.append(conv2d(conv2, filters=self.num_classes, kernel_size=3,
                                  padding='same'))
            deconvs.append(self._upsample(bn2, k=1))
            deconvs.append(self._upsample(bn4, k=2))
            deconvs.append(self._upsample(d1, k=3))
            deconvs.append(self._upsample(d2, k=4))
            deconvs.append(self._upsample(d3, k=5))

            concat = tf.concat(deconvs, axis=3)

            conv3 = conv2d(concat, filters=self.num_classes, kernel_size=3, padding='same')
            ac1 = self._add_common_layers(conv3)

            conv4 = conv2d(ac1, filters=self.num_classes, kernel_size=1, padding='same')
            ac2 = self._add_common_layers(conv4)

            self.preds_seg = softmax(ac2)

        with tf.variable_scope("decoder_cont"):
            deconvs = []
            deconvs.append(conv2d(conv2, filters=self.num_classes, kernel_size=3,
                                  padding='same'))
            deconvs.append(self._upsample(bn2, k=1))
            deconvs.append(self._upsample(bn4, k=2))
            deconvs.append(self._upsample(d1, k=3))
            deconvs.append(self._upsample(d2, k=4))
            deconvs.append(self._upsample(d3, k=5))

            concat = tf.concat(deconvs, axis=3)

            conv3 = conv2d(concat, filters=self.num_classes, kernel_size=3, padding='same')
            ac1 = self._add_common_layers(conv3)

            conv4 = conv2d(ac1, filters=self.num_classes, kernel_size=1, padding='same')
            ac2 = self._add_common_layers(conv4)

            self.preds_cont = softmax(ac2)

        cond1 = tf.greater_equal(self.preds_seg, 0.5)
        cond2 = tf.less(self.preds_cont, 0.5)

        conditions = tf.logical_and(cond1, cond2)

        self.preds = tf.where(conditions, tf.ones_like(conditions), tf.zeros_like(conditions))

        self._add_train_op()

        self.summaries = tf.summary.merge_all()
