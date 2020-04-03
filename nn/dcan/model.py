import tensorflow.compat.v1 as tf
import numpy as np
from tensorflow.compat.v1.nn import softmax
from tensorflow.compat.v1.layers import batch_normalization, dropout, max_pooling2d, conv2d, conv2d_transpose
#from metrics import hausdorff_object_score, dice_object_score, object_f_score

tf.disable_eager_execution()
__author__ = "Mathias Baltzersen and Rasmus Hvingelby"

class DCAN:
    def __init__(self):

        self.num_classes = 3
        self.img_input_channels = 1

        self.threshold = tf.constant(0.5, dtype=tf.float32)
        self.l2_scale = 0
        self.contour_loss_weight = 1

        self.dropout_p = 0.5
        self.learning_rate_value = 0.005

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
                variable_parameters *= dim
            total_parameters += variable_parameters
        print("Total parameters of model.: {}".format(total_parameters))

    def train(self, train_generators, epochs, output):
        summary_writer = tf.summary.FileWriter(output)

        for epoch in range(epochs):
            step = self._train_one_epoch(train_generators, summary_writer)
            
            if epoch % 10 == 0:
                self.saver.save(self.sess, output + '/model', global_step=step)

        self.saver.save(self.sess, output + '/final-model')


    def _train_one_epoch(self, generators, summary_writer):
        
        for img_batch, mask_batch, contours_batch in generators:

            feed_dict = {
                self.input_image: img_batch,
                self.gt_image: mask_batch,
                self.gt_contours: contours_batch,
                self.dropout_prob: self.dropout_p,
                self.lr: self.learning_rate_value
            }

            _, summary, step, loss = self.sess.run([self.train_op, self.summaries, self.global_step, self.loss],
                                                   feed_dict=feed_dict)

            if step > 10000:
                self.learning_rate_value = 0.00005

            summary_writer.add_summary(summary, step)

        return step

    def predict(self, images, batch_size=32):
        num_examples = images.shape[0]
        num_batches = np.ceil(num_examples // batch_size)

        batches = np.array_split(images, num_batches)

        ## TODO : Revenir si kkchose marche pas
        dropout_probability = 0.0
        batch_predictions = []

        for input_image_batch in batches:
            feed_dict = {
                self.input_image: input_image_batch,
                self.dropout_prob: dropout_probability
            }

            output_predictions = self.sess.run(self.preds, feed_dict=feed_dict)

            batch_predictions.extend(output_predictions)

        return np.array(batch_predictions)


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
        bn = batch_normalization(inputs)
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

            variables = tf.trainable_variables()

            #Apply regularization to all non bias variables
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in variables
                               if 'bias' not in v.name]) * self.l2_scale

            loss = tf.add_n([lossL2, seg_loss, cont_loss])

            self.loss = tf.reduce_mean(loss, name='loss')

        tf.summary.scalar('loss', self.loss)

        #TODO: In DCAN they use SGD.
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    def _create_model(self):

        self.input_image = tf.placeholder(tf.float32, shape=(None, 256, 256, self.img_input_channels), name='input_image_placeholder')
        self.gt_image = tf.placeholder(tf.int32, shape=(None, 256, 256, self.num_classes), name='gt_image_placeholder')
        self.gt_contours = tf.placeholder(tf.int32, shape=(None, 256, 256, self.num_classes), name='gt_contours_placeholder')
        self.dropout_prob = tf.placeholder(dtype=tf.float32, shape=None, name='dropout_prob_placeholder')

        self.lr = tf.placeholder(dtype=tf.float32, shape=None, name='learning_rate_placeholder')

        scale_nc = 1

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
