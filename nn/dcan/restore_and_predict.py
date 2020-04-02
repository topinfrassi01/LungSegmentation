import tensorflow as tf
import numpy as np
from PIL import Image

from image_segmenter.preprocess import GlandHandler

__author__ = "Rasmus Hvingelby"

sess = tf.Session()

folder = "/home/hvingelby/Workspace/medical_image_seg/SpecialCourse_MedicalImageSeg/image_segmenter/some_model"

latest_model_path = tf.train.latest_checkpoint(folder)

saver = tf.train.import_meta_graph('/home/hvingelby/Workspace/medical_image_seg/SpecialCourse_MedicalImageSeg/image_segmenter/some_model/model-10000.meta')
saver.restore(sess, latest_model_path)

graph = tf.get_default_graph()

#print([n.name for n in tf.get_default_graph().as_graph_def().node])

input_image = graph.get_tensor_by_name("Placeholder:0") #input image (bs, w, h, channels)
placeholder_1 = graph.get_tensor_by_name("Placeholder_1:0") #ground truth image (bs, w, h)
dropout_prob = graph.get_tensor_by_name("Placeholder_2:0") #dropoutprob float
placeholder_3 = graph.get_tensor_by_name("Placeholder_3:0") #learning rate float

logits = graph.get_operation_by_name('decoder/Relu_16')
softmax = graph.get_tensor_by_name('decoder/softmax/Reshape_1:0')


batch_size = 2
data_path = '/home/hvingelby/Workspace/medical_image_seg/gland'

x_train, y_train, x_a_test, y_a_test, x_b_test, y_b_test = GlandHandler(data_path).get_gland()

input_images = x_train

num_examples = input_images.shape[0]
num_batches = np.ceil(num_examples / batch_size)

input_image_batches = np.array_split(input_images, num_batches)

for i, input_image_batch in enumerate(input_image_batches):
    print("Batch: {}".format(i))
    feed_dict = {
        input_image: input_image_batch,
        dropout_prob: 1.0
    }

    output_predictions = sess.run(softmax, feed_dict=feed_dict)

    for j in range(output_predictions.shape[0]):
        img = (np.argmax(output_predictions[j], axis=2) * 255).astype(np.uint8)
        img = Image.fromarray(img, mode='L')
        img.save('./my_preds/' + str(i)+str(j) + '.bmp')