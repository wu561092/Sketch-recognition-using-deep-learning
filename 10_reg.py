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

ITERATION=20000


BATCH_SIZE=64


net_data = load('bvlc_alexnet.npy').item()

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

def conv_relu(input, kernel_shape, bias_shape,stride):
# Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape,
        initializer=tf.random_normal_initializer())
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape,
        initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights,
        strides=[1, stride, stride, 1], padding='SAME')
    return tf.nn.relu(conv + biases)

def fc_layer( bottom, n_weight, name):
    with tf.name_scope(name) as scope:
        n_prev_weight = bottom.get_shape()[1]
        initer = tf.truncated_normal_initializer(stddev=0.01)
        weight_decay = tf.constant(0.0005, dtype=tf.float32) # your weight decay rate, must be a scalar tensor.
        W = tf.get_variable(name+'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=initer,regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
        b = tf.get_variable(name+'b', dtype=tf.float32, initializer=tf.constant(0.01, shape=[n_weight], dtype=tf.float32))
        fc = tf.nn.bias_add(tf.matmul(bottom, W), b)
    return fc
#new_book
def mpool_op(input_op,name,kh,kw,dh,dw):
    return tf.nn.max_pool(input_op,ksize=[1,kh,kh,1],strides=[1,dh,dw,1],padding='SAME')

def s_net(input_op,keep_prob):
    p=[]
    with tf.variable_scope("s_conv1"):
        conv1W = tf.Variable(net_data["conv1"][0],name="s_conv1w")
        conv1b = tf.Variable(net_data["conv1"][1],name="s_conv1b")
        conv1_in = conv(input_op, conv1W, conv1b, 11, 11, 96, 4, 4, padding="SAME", group=1)
        conv1 = tf.nn.relu(conv1_in)
    with tf.variable_scope("s_pool1"):
        maxpool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID',name="s_poo1")
    with tf.variable_scope("s_conv2"):
        conv2W = tf.Variable(net_data["conv2"][0],name="s_conv2w")
        conv2b = tf.Variable(net_data["conv2"][1],name="s_conv2b")
        conv2_in = conv(maxpool1, conv2W, conv2b,5, 5, 256, 1, 1, padding="SAME", group=2)
        conv2 = tf.nn.relu(conv2_in)
    with tf.variable_scope("s_pool2"):
        maxpool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID',name="s_poo2")
    with tf.variable_scope("s_conv3"):
        conv3W = tf.Variable(net_data["conv3"][0],name="s_conv3w")
        conv3b = tf.Variable(net_data["conv3"][1],name="s_conv3b")
        conv3_in = conv(maxpool2, conv3W, conv3b, 3, 3, 384, 1, 1, padding="SAME", group=1)
        conv3 = tf.nn.relu(conv3_in)
    return conv3

def pr_net(input_op,keep_prob):
    p=[]
    with tf.variable_scope("p_conv1"):
        conv1W = tf.Variable(net_data["conv1"][0],name="p_conv1w")
        conv1b = tf.Variable(net_data["conv1"][1],name="p_conv1b")
        conv1_in = conv(input_op, conv1W, conv1b, 11, 11, 96, 4, 4, padding="SAME", group=1)
        conv1 = tf.nn.relu(conv1_in)
    with tf.variable_scope("p_pool1"):
        maxpool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID',name="p_pool1")
    with tf.variable_scope("p_conv2"):
        conv2W = tf.Variable(net_data["conv2"][0],name="p_conv2w")
        conv2b = tf.Variable(net_data["conv2"][1],name="p_conv2b")
        conv2_in = conv(maxpool1, conv2W, conv2b,5, 5, 256, 1, 1, padding="SAME", group=2)
        conv2 = tf.nn.relu(conv2_in)
    with tf.variable_scope("p_pool2"):
        maxpool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID',name="p_pool2")
    with tf.variable_scope("p_conv3"):
        conv3W = tf.Variable(net_data["conv3"][0],name="p_conv3w")
        conv3b = tf.Variable(net_data["conv3"][1],name="p_conv3b")
        conv3_in = conv(maxpool2, conv3W, conv3b, 3, 3, 384, 1, 1, padding="SAME", group=1)
        conv3 = tf.nn.relu(conv3_in)
    with tf.variable_scope("p_conv4"):
        conv4W = tf.Variable(net_data["conv4"][0],name="p_conv4w")
        conv4b = tf.Variable(net_data["conv4"][1],name="p_conv4b")
        conv4_in = conv(conv3, conv4W, conv4b, 3, 3, 384, 1, 1, padding="SAME", group=2)
        conv4 = tf.nn.relu(conv4_in)

    return conv4

def nr_net(input_op,keep_prob):
    p=[]
    with tf.variable_scope("n_conv1"):
        conv1W = tf.Variable(net_data["conv1"][0],name="n_conv1w")
        conv1b = tf.Variable(net_data["conv1"][1],name="n_conv1b")
        conv1_in = conv(input_op, conv1W, conv1b, 11, 11, 96, 4, 4, padding="SAME", group=1)
        conv1 = tf.nn.relu(conv1_in)
    with tf.variable_scope("n_pool1"):
        maxpool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID',name="n_pool1")
    with tf.variable_scope("n_conv2"):
        conv2W = tf.Variable(net_data["conv2"][0],name="n_conv2w")
        conv2b = tf.Variable(net_data["conv2"][1],name="n_conv2b")
        conv2_in = conv(maxpool1, conv2W, conv2b,5, 5, 256, 1, 1, padding="SAME", group=2)
        conv2 = tf.nn.relu(conv2_in)
    with tf.variable_scope("n_pool2"):
        maxpool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID',name="n_pool2")
    with tf.variable_scope("n_conv3"):
        conv3W = tf.Variable(net_data["conv3"][0],name="n_conv3w")
        conv3b = tf.Variable(net_data["conv3"][1],name="n_conv3b")
        conv3_in = conv(maxpool2, conv3W, conv3b, 3, 3, 384, 1, 1, padding="SAME", group=1)
        conv3 = tf.nn.relu(conv3_in)
    with tf.variable_scope("n_conv4"):
        conv4W = tf.Variable(net_data["conv4"][0],name="n_conv4w")
        conv4b = tf.Variable(net_data["conv4"][1],name="n_conv4b")
        conv4_in = conv(conv3, conv4W, conv4b, 3, 3, 384, 1, 1, padding="SAME", group=2)
        conv4 = tf.nn.relu(conv4_in)

    return conv4

def c_net(input_op):
    pp=[]
    with tf.variable_scope("conv1"):
        conv1 = conv_relu(input_op, [3, 3, 768, 384], [384],1)
    with tf.variable_scope("conv2"):
        conv2= conv_relu(conv1, [3, 3, 384, 384], [384],1)
    with tf.variable_scope("conv3"):
        conv3= conv_relu(conv2, [3, 3, 384, 256], [256],1)
    with tf.variable_scope("pool1"):
        pool1=mpool_op(conv3,name="pool1",kh=3,kw=3,dh=2,dw=2)
    shp = pool1.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool1, [-1, flattened_shape], name="c_poo1_reshape")
    fc1 = fc_layer(resh1, 512 , "fc1")
    fc2 = fc_layer(fc1, 512 , "fc2")
    fc3 = fc_layer(fc2,  NUM_CLASSES, "fc3")
    return fc3

def custom_loss(o1_soft,o2_soft):
    with tf.name_scope("Lr_loss"):
        o1_np=np.asarray(o1_soft)
        o2_np=np.asarray(o2_soft)
        eve_batch_loss=np.sum(np.abs(o1_np-o2_np))
        alllose=tf.reduce_mean(eve_batch_loss)
        losses=tf.maximum(0.0,1-alllose)
    return losses

def loss_with_spring(o1,o2,y2_,y3_):
    with tf.name_scope("Lc_loss"):
        o1_soft = tf.nn.softmax(o1)
        o1_softcross = -tf.reduce_sum(y2_*tf.log(tf.clip_by_value(o1_soft,1e-10,1.0)))
        o2_soft = tf.nn.softmax(o2)
        o2_softcross = -tf.reduce_sum(y3_*tf.log(tf.clip_by_value(o2_soft,1e-10,1.0)))
        #o1_softcross=tf.nn.softmax_cross_entropy_with_logits(logits=o1,labels=y2_)
        #o2_softcross=tf.nn.softmax_cross_entropy_with_logits(logits=o2,labels=y3_)
          # Choose an appropriate one.
        lc_loss =tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        lc1_cross_entropy=tf.reduce_mean(o1_softcross)
        lc2_cross_entropy=tf.reduce_mean(o2_softcross)
    return o1_softcross,o2_softcross,lc1_cross_entropy,lc2_cross_entropy,lc_loss


def main():

    tf.reset_default_graph()

    image, label = read_and_decode("sketch_train2.tfrecords")
    posimage, poslabel = real_read_and_decode("pos_train.tfrecords")
    negimage, neglabel = real_read_and_decode("neg_train3.tfrecords")

    image_batch, label_batch =tf.train.batch([image, label],batch_size=BATCH_SIZE)
    posimage_batch, poslabel_batch = tf.train.batch([posimage, poslabel],batch_size=BATCH_SIZE)
    negimage_batch, neglabel_batch = tf.train.batch([negimage, neglabel],batch_size=BATCH_SIZE)

    keep_prob=tf.placeholder(tf.float32)

    with tf.name_scope('sketch_input'):     #ML hw3_example code
        x1 = tf.placeholder(tf.float32, [None, WIDTH, HEIGHT, 3],name="sketch_input")
        y1 = tf.placeholder(tf.float32, [None, NUM_CLASSES],name="sketch_label")
    with tf.name_scope('positive_input'):
        x2 = tf.placeholder(tf.float32, [None, WIDTH, HEIGHT, 3],name="positive_real_input")
        y2 = tf.placeholder(tf.float32, [None, NUM_CLASSES],name="positive_label")
    with tf.name_scope('nagative_input'):
        x3 = tf.placeholder(tf.float32, [None, WIDTH, HEIGHT, 3],name="negative_real_input")
        y3 = tf.placeholder(tf.float32, [None, NUM_CLASSES],name="negative_label")

#    with tf.device('/gpu:1'):
    s_conv3=s_net(x1, keep_prob)
 #   with tf.device('/gpu:2'):
    pr_conv4=pr_net(x2, keep_prob)
  # with tf.device('/gpu:3'):
    nr_conv4=nr_net(x3, keep_prob)

    p_input=tf.concat([pr_conv4,s_conv3],name="positive_input",axis=3)
    n_input=tf.concat([nr_conv4,s_conv3],name="negative_input",axis=3)


    with tf.variable_scope("siamese") as scope:
        p_fc3 = c_net(p_input)
        scope.reuse_variables()
        n_fc3 = c_net(n_input)


    o1_softcross,o2_softcross,lc1_cross_entropy,lc2_cross_entropy,lc_loss= loss_with_spring(p_fc3,n_fc3,y2,y3)
    lr_loss = custom_loss(o1_softcross,o2_softcross)



    with tf.name_scope("loss"):
        loss=lr_loss+lc_loss
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('lr_loss',lr_loss)
        tf.summary.scalar('lc1_loss',lc1_cross_entropy)
        tf.summary.scalar('lc2_loss',lc2_cross_entropy)


    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.0001
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           2000, 0.9, staircase=True)
# Passing global_step to minimize() will increment it at each step.

    with tf.name_scope("train_step"):
        train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate , momentum=0.9).minimize(loss,global_step=global_step)
    # Create loss


    sess=tf.Session()
    merged = tf.summary.merge_all()
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter('./loss_reg_graphs_20000',sess.graph)

    with tf.Session(config=config) as sess:
        try:
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess,coord=coord)

            for i in range(ITERATION):
                if coord.should_stop():
    				print('corrd break!!!!!!')
    				break
                #sess.run(tf.local_variables_initializer())
                example_train, l_train = sess.run([image_batch, label_batch])

                example_train2, l_train2= sess.run([posimage_batch, poslabel_batch])

                example_train3, l_train3 = sess.run([negimage_batch, neglabel_batch])
                _, loss_v = sess.run([train_step, loss],
                                feed_dict={x1: example_train,y1: l_train,
                                x2: example_train2, y2: l_train2,
                                x3: example_train3, y3: l_train3})
                if (i % 10 == 0) & (i != 0):
                    # load test data batch
                    result = sess.run(merged,feed_dict={x1: example_train,y1: l_train,
                                x2: example_train2, y2: l_train2,
                                x3: example_train3, y3: l_train3})
                    writer.add_summary(result, i)
                    # test accuracy

                    print('iter: ', i)
                    print('loss: ', loss_v)


        except tf.errors.OutOfRangeError:
            print('Done training -- limit reached')

        save_path = saver.save(sess, "./model_reg_20000/model.ckpt")

        coord.request_stop()
        coord.join(threads)

        writer.close()





main()
