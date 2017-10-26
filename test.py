import tensorflow as tf
import numpy as np
import os
import glob
import tensorflow as tf
from PIL import Image
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from numpy import *
import os
import time
import urllib
from numpy import random


config=tf.ConfigProto()
config.gpu_options.allow_growth=True






NUM_CLASSES = 10 # each class train data number
WIDTH = 227
HEIGHT = 227
CHANNELS = 3

ITERATION=2000

WEIGHT_DECAY_FACTOR=0.0005
BATCH_SIZE=64


sess=tf.Session()
saver = tf.train.import_meta_graph('./model/model.ckpt.meta')
saver.restore(sess,tf.train.latest_checkpoint('./model/'))
graph = tf.get_default_graph()





# Internet TFrecords
def read_and_decode(filename):
#    print filename
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [WIDTH,HEIGHT, 3])
    label = tf.cast(features['label'], tf.int32)
    labels = tf.one_hot(label, NUM_CLASSES)
    #print img,labels
    return img, labels

def real_read_and_decode(filename):
#    print filename
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [WIDTH,HEIGHT, 3])
    label = tf.cast(features['label'], tf.int32)
    labels = tf.one_hot(label, NUM_CLASSES)
    #print img,labels
    return img, labels

def conv_relu(input, weights, biases,stride):
# Create variable named "weights".

    conv = tf.nn.conv2d(input, weights,
        strides=[1, stride, stride, 1], padding='SAME')
    return tf.nn.relu(conv + biases)


#new_book
def mpool_op(input_op,name,kh,kw,dh,dw):
    return tf.nn.max_pool(input_op,ksize=[1,kh,kh,1],strides=[1,dh,dw,1],padding='SAME')


def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):

    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)


    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(input, group, 3)
        kernel_groups = tf.split(kernel,group, 3)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups,3)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])

def s_net(input_op,keep_prob):
    p=[]

    s_conv1W = graph.get_tensor_by_name(name="s_conv1/s_conv1w:0")
    s_conv1b = graph.get_tensor_by_name(name="s_conv1/s_conv1b:0")
    #s_conv1 = conv_relu(input_op, s_conv1W, s_conv1b,4)
    s_conv1_in = conv(input_op, s_conv1W, s_conv1b, 11, 11, 96, 4, 4, padding="SAME", group=1)
    s_conv1 = tf.nn.relu(s_conv1_in)

    s_maxpool1 = tf.nn.max_pool(s_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID',name="s_poo1")

    s_conv2W =graph.get_tensor_by_name(name="s_conv2/s_conv2w:0")
    s_conv2b =graph.get_tensor_by_name(name="s_conv2/s_conv2b:0")
    #s_conv2 = conv_relu(s_maxpool1, s_conv2W, s_conv2b,1)
    s_conv2_in = conv(s_maxpool1,s_conv2W,s_conv2b,5, 5, 256, 1, 1, padding="SAME", group=2)
    s_conv2 = tf.nn.relu(s_conv2_in)


    s_maxpool2 = tf.nn.max_pool(s_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID',name="s_poo2")

    s_conv3W =graph.get_tensor_by_name(name="s_conv3/s_conv3w:0")
    s_conv3b =graph.get_tensor_by_name(name="s_conv3/s_conv3b:0")
    s_conv3 = conv_relu(s_maxpool2, s_conv3W, s_conv3b,1)
    s_conv3_in = conv(s_maxpool2, s_conv3W, s_conv3b, 3, 3, 384, 1, 1, padding="SAME", group=1)
    s_conv3 = tf.nn.relu(s_conv3_in)

    return s_conv3


def nr_net(input_op,keep_prob):
    p=[]
    n_conv1W = graph.get_tensor_by_name(name="n_conv1/n_conv1w:0")
    n_conv1b = graph.get_tensor_by_name(name="n_conv1/n_conv1b:0")
    n_conv1_in = conv(input_op,n_conv1W,n_conv1b, 11, 11, 96, 4, 4, padding="SAME", group=1)
    n_conv1 = tf.nn.relu(n_conv1_in)

    n_maxpool1 = tf.nn.max_pool(n_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID',name="n_pool1")

    n_conv2W = graph.get_tensor_by_name(name="n_conv2/n_conv2w:0")
    n_conv2b = graph.get_tensor_by_name(name="n_conv2/n_conv2b:0")
    n_conv2_in = conv(n_maxpool1,n_conv2W,n_conv2b,5, 5, 256, 1, 1, padding="SAME", group=2)
    n_conv2 = tf.nn.relu(n_conv2_in)

    n_maxpool2 = tf.nn.max_pool(n_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID',name="n_pool2")

    n_conv3W =graph.get_tensor_by_name(name="n_conv3/n_conv3w:0")
    n_conv3b = graph.get_tensor_by_name(name="n_conv3/n_conv3b:0")
    n_conv3_in = conv(n_maxpool2,n_conv3W,n_conv3b, 3, 3, 384, 1, 1, padding="SAME", group=1)
    n_conv3 = tf.nn.relu(n_conv3_in)

    n_conv4W = graph.get_tensor_by_name(name="n_conv4/n_conv4w:0")
    n_conv4b = graph.get_tensor_by_name(name="n_conv4/n_conv4b:0")
    n_conv4_in = conv(n_conv3,n_conv4W,n_conv4b, 3, 3, 384, 1, 1, padding="SAME", group=2)
    n_conv4 = tf.nn.relu(n_conv4_in)


    return n_conv4

def c_net(input_op):

    c_conv1W = graph.get_tensor_by_name(name="siamese/conv1/weights:0 ")
    c_conv1b = graph.get_tensor_by_name(name="siamese/conv1/biases:0")
    c_conv1 = conv_relu(input_op, c_conv1W, c_conv1b,1)

    c_conv2W = graph.get_tensor_by_name(name="siamese/conv2/weights:0")
    c_conv2b = graph.get_tensor_by_name(name="siamese/conv2/biases:0")
    c_conv2 = conv_relu(c_conv1, c_conv2W, c_conv2b,1)

    c_conv3W = graph.get_tensor_by_name(name="siamese/conv3/weights:0 ")
    c_conv3b = graph.get_tensor_by_name(name="siamese/conv3/biases:0")
    c_conv3 = conv_relu(c_conv2, c_conv3W, c_conv3b,1)

    pool1=mpool_op(c_conv3,name="pool1",kh=3,kw=3,dh=2,dw=2)

    shp = pool1.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool1, [-1, flattened_shape], name="c_poo1_reshape")

    c_fc1W = graph.get_tensor_by_name(name="siamese/fc1W:0")
    c_fc1b = graph.get_tensor_by_name(name="siamese/fc1b:0")
    c_fc1 =  tf.nn.relu(tf.matmul(resh1, c_fc1W) + 	c_fc1b)

    c_fc2W = graph.get_tensor_by_name(name="siamese/fc2W:0")
    c_fc2b = graph.get_tensor_by_name(name="siamese/fc2b:0")
    c_fc2 =  tf.nn.relu(tf.matmul(c_fc1, c_fc2W) + 	c_fc2b)

    c_fc3W = graph.get_tensor_by_name(name="siamese/fc3W:0")
    c_fc3b = graph.get_tensor_by_name(name="siamese/fc3b:0")
    c_fc3 =  tf.matmul(c_fc2, c_fc3W) + c_fc3b

    prediction = tf.nn.softmax(c_fc3 )

    return prediction




def main():



    image, label = read_and_decode("sketch_train2.tfrecords")
    negimage, neglabel = real_read_and_decode("neg_train3.tfrecords")

    image_batch, label_batch =tf.train.batch([image, label],batch_size=BATCH_SIZE)
    negimage_batch, neglabel_batch = tf.train.batch([negimage, neglabel],batch_size=BATCH_SIZE)

    keep_prob=tf.placeholder(tf.float32)

    with tf.name_scope('sketch_input'):     #ML hw3_example code
        x1 = tf.placeholder(tf.float32, [None, WIDTH, HEIGHT, 3],name="sketch_input")
        y1 = tf.placeholder(tf.float32, [None, NUM_CLASSES],name="sketch_label")
    with tf.name_scope('nagative_input'):
        x3 = tf.placeholder(tf.float32, [None, WIDTH, HEIGHT, 3],name="negative_real_input")
        y3 = tf.placeholder(tf.float32, [None, NUM_CLASSES],name="negative_label")



    s_conv3=s_net(x1, 0)

    nr_conv4=nr_net(x3,0)


    n_input=tf.concat([nr_conv4,s_conv3],name="negative_input",axis=3)

    y_prediction =c_net(n_input)
    print y_prediction

    loss = tf.reduce_sum(tf.losses.sigmoid_cross_entropy(y1, y_prediction)) # loss
    train_step = tf.train.AdamOptimizer(1e-3).minimize(loss) # optimizer#
	# accuracy
    correct = tf.equal(tf.argmax(y_prediction, 1), tf.argmax(y1, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    # Create loss



    sess=tf.Session()

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./test_graphs1',sess.graph)

    with tf.Session(config=config) as sess:
        try:
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess,coord=coord)


            for i in range(ITERATION):
                if coord.should_stop():
    				print('corrd break!!!!!!')
    				break
                #sess.run(tf.local_variables_initializer())
                example_train, l_train = sess.run([image_batch, label_batch])
                example_train3, l_train3 = sess.run([negimage_batch, neglabel_batch])

                _, loss_v = sess.run([train_step, loss],
                                feed_dict={x1: example_train,y1: l_train,
                                x3: example_train3, y3: l_train3})
                if (i % 10 == 0) & (i != 0):
                    # load test data batch
				# test accuracy
			        test_acc = sess.run(accuracy,feed_dict={x1: example_train,y1: l_train,
                            x3: example_train3, y3: l_train3})
				print('iter: ', i)
				print('loss: ', loss_v)
				print('test_acc: ', test_acc)




        except tf.errors.OutOfRangeError:
            print('Done training -- limit reached')



        coord.request_stop()
        coord.join(threads)

        writer.close()





main()
