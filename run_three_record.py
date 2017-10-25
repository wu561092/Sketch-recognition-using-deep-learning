import tensorflow as tf
import numpy as np
import os
import glob
import tensorflow as tf
from PIL import Image
from tensorflow.python.framework.graph_util import convert_variables_to_constants



config=tf.ConfigProto()
config.gpu_options.allow_growth=True



learning_rate = 0.001
batch_size = 2





NUM_CLASSES = 10 # each class train data number
WIDTH = 227
HEIGHT = 227
CHANNELS = 1

ITERATION=10000

WEIGHT_DECAY_FACTOR=0.0005
BATCH_SIZE=200





# Internet TFrecords
def read_and_decode(filename):
    print filename
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })
    img= tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [WIDTH,HEIGHT, 3])

    label = tf.cast(features['label'], tf.int32)
    labels = tf.one_hot(label, NUM_CLASSES)
    print img,labels
    return img, labels




def main():


    image, label = read_and_decode("sketch_train2.tfrecords")
    posimage, poslabel = read_and_decode("pos_train.tfrecords")
    negimage, neglabel = read_and_decode("neg_train3.tfrecords")

    image_batch, label_batch = tf.train.shuffle_batch([image, label],batch_size=BATCH_SIZE,capacity=500, min_after_dequeue=100)
    posimage_batch, poslabel_batch = tf.train.shuffle_batch([posimage, poslabel],batch_size=BATCH_SIZE,capacity=500, min_after_dequeue=100)
    negimage_batch, neglabel_batch = tf.train.shuffle_batch([negimage, neglabel],batch_size=BATCH_SIZE,capacity=500, min_after_dequeue=100)

    sess=tf.Session()



    with tf.Session() as sess:
        try:
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)


            for i in range(ITERATION):
                example_train, l_train = sess.run([image_batch, label_batch])
                example_train2, l_train2= sess.run([posimage_batch, poslabel_batch])
                example_train3, l_train3 = sess.run([negimage_batch, neglabel_batch])
                print i
                print "s+"
                print example_train.shape, l_train
                print "p+"
                print example_train2.shape, l_train2
                print "n+"
                print example_train3.shape, l_train3

        except tf.errors.OutOfRangeError:
            print('Done training -- limit reached')

        coord.request_stop()
        coord.join(threads)




main()
