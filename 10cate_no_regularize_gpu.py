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






NUM_CLASSES = 10 # each class train data number
WIDTH = 227
HEIGHT = 227
CHANNELS = 3

ITERATION=10000

WEIGHT_DECAY_FACTOR=0.0005
BATCH_SIZE=30





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
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [WIDTH,HEIGHT, 3])
    label = tf.cast(features['label'], tf.int32)
    labels = tf.one_hot(label, NUM_CLASSES)
    #print img,labels
    return img, labels

def real_read_and_decode(filename):
    print filename
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

def conv_op(input_op,name,kh,kw,n_out,dh,dw,p):
    n_in=input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel=tf.get_variable(scope+"w",shape=[kh,kw,n_in,n_out],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv=tf.nn.conv2d(input_op,kernel,(1,dh,dw,1),padding='SAME')
        bias_init_val=tf.constant(0.0,shape=[n_out],dtype=tf.float32)
        biases=tf.Variable(bias_init_val,trainable=True,name='b')
        z=tf.nn.bias_add(conv,biases)
        activation=tf.nn.relu(z,name=scope)
        p+=[kernel,biases]
        return activation

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

def fc_op(input_op, name, n_out, p):
    n_in=input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel=tf.get_variable(scope+"w",shape=[n_in,n_out],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases=tf.Variable(tf.constant(0.1,shape=[n_out],dtype=tf.float32),name='b')
        activation=tf.nn.relu_layer(input_op,kernel,biases,name=scope)
        p+=[kernel,biases]
        return activation

def fc_layer( bottom, n_weight, name):
    with tf.name_scope(name) as scope:
        n_prev_weight = bottom.get_shape()[1]
        initer = tf.truncated_normal_initializer(stddev=0.01)
        W = tf.get_variable(name+'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=initer)
        b = tf.get_variable(name+'b', dtype=tf.float32, initializer=tf.constant(0.01, shape=[n_weight], dtype=tf.float32))
        fc = tf.nn.bias_add(tf.matmul(bottom, W), b)
    return fc
#new_book
def mpool_op(input_op,name,kh,kw,dh,dw):
    return tf.nn.max_pool(input_op,ksize=[1,kh,kh,1],strides=[1,dh,dw,1],padding='SAME')

def s_net(input_op,keep_prob):
    p=[]
    conv1=conv_op(input_op,name="s_conv1",kh=11,kw=11,n_out=96,dh=4,dw=4,p=p)
    with tf.variable_scope("s_pool1"):
        pool1=mpool_op(conv1,name="s_pool1",kh=3,kw=3,dh=2,dw=2)
    conv2=conv_op(pool1,name="s_conv2",kh=5,kw=5,n_out=256,dh=1,dw=1,p=p)
    with tf.variable_scope("s_pool2"):
        pool2=mpool_op(conv2,name="s_pool2",kh=3,kw=3,dh=2,dw=2)
    conv3=conv_op(pool2,name="s_out_conv3",kh=3,kw=3,n_out=384,dh=1,dw=1,p=p)
    return conv3

def pr_net(input_op,keep_prob):
    p=[]
    conv1=conv_op(input_op,name="p_conv1",kh=11,kw=11,n_out=96,dh=4,dw=4,p=p)
    with tf.variable_scope("p_pool1"):
        pool1=mpool_op(conv1,name="p_pool1",kh=3,kw=3,dh=2,dw=2)
    conv2=conv_op(pool1,name="p_conv2",kh=5,kw=5,n_out=256,dh=1,dw=1,p=p)
    with tf.variable_scope("p_pool2"):
        pool2=mpool_op(conv2,name="p_pool2",kh=3,kw=3,dh=2,dw=2)
    conv3=conv_op(pool2,name="p_conv3",kh=3,kw=3,n_out=384,dh=1,dw=1,p=p)
    conv4=conv_op(conv3,name="p_conv4",kh=3,kw=3,n_out=384,dh=1,dw=1,p=p)
    return conv4

def nr_net(input_op,keep_prob):
    p=[]
    conv1=conv_op(input_op,name="n_conv1",kh=11,kw=11,n_out=96,dh=4,dw=4,p=p)
    with tf.variable_scope("n_pool1"):
        pool1=mpool_op(conv1,name="n_pool1",kh=3,kw=3,dh=2,dw=2)
    conv2=conv_op(pool1,name="n_conv2",kh=5,kw=5,n_out=256,dh=1,dw=1,p=p)
    with tf.variable_scope("n_pool2"):
        pool2=mpool_op(conv2,name="n_pool2",kh=3,kw=3,dh=2,dw=2)
    conv3=conv_op(pool2,name="n_conv3",kh=3,kw=3,n_out=384,dh=1,dw=1,p=p)
    conv4=conv_op(conv3,name="n_conv4",kh=3,kw=3,n_out=384,dh=1,dw=1,p=p)
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
    fc2 = fc_layer(fc1,  NUM_CLASSES, "fc2")
    return fc2


def custom_loss(o1_soft,o2_soft):
    with tf.name_scope("Lr_loss"):
        o1_np=np.asarray(o1_soft)
        o2_np=np.asarray(o2_soft)
        eve_batch_loss=np.sum(np.abs(o1_np-o2_np))
        alllose=tf.reduce_mean(eve_batch_loss)
        losses=tf.maximum(0.0,1-alllose)
    return losses

def loss_with_spring(o1,o2,y1_,y2_):
    with tf.name_scope("Lc_loss"):
        labels_t = y1_
        labels_t = y2_
        o1_soft=tf.nn.softmax(o1)
        o2_soft=tf.nn.softmax(o2)
        o1_softcross=tf.nn.softmax_cross_entropy_with_logits(logits=o1_soft,labels=y1_)
        o2_softcross=tf.nn.softmax_cross_entropy_with_logits(logits=o2_soft,labels=y2_)
        lc_cross_entropy=tf.reduce_mean(o1_softcross+o2_softcross)
    return o1_softcross,o2_softcross,lc_cross_entropy



def main():

    image, label = read_and_decode("sketch_train2.tfrecords")
    posimage, poslabel = real_read_and_decode("pos_train.tfrecords")
    negimage, neglabel = real_read_and_decode("neg_train3.tfrecords")

    image_batch, label_batch =tf.train.shuffle_batch([image, label],batch_size=BATCH_SIZE,capacity=1500, min_after_dequeue=1000)
    posimage_batch, poslabel_batch = tf.train.shuffle_batch([posimage, poslabel],batch_size=BATCH_SIZE,capacity=1500, min_after_dequeue=1000)
    negimage_batch, neglabel_batch = tf.train.shuffle_batch([negimage, neglabel],batch_size=BATCH_SIZE,capacity=1500, min_after_dequeue=1000)

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

    with tf.device('/gpu:1'):
        s_conv3=s_net(x1, keep_prob)
    with tf.device('/gpu:2'):
        pr_conv4=pr_net(x2, keep_prob)
    with tf.device('/gpu:3'):
        nr_conv4=nr_net(x3, keep_prob)

    p_input=tf.concat([pr_conv4,s_conv3],name="positive_input",axis=3)
    n_input=tf.concat([nr_conv4,s_conv3],name="negative_input",axis=3)


    with tf.variable_scope("siamese") as scope:
        temp1 = c_net(p_input)
        scope.reuse_variables()
        temp2 = c_net(n_input)


    o1_softcross,o2_softcross,lc_cross_entropy= loss_with_spring(temp1,temp2,y2,y3)
    lr_loss = custom_loss(o1_softcross,o2_softcross)



    with tf.name_scope("loss"):
        loss=lr_loss+lc_cross_entropy
        tf.summary.scalar('loss', loss)
    with tf.name_scope("train_step"):
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    # Create loss


    sess=tf.Session()
    merged = tf.summary.merge_all()

    writer = tf.summary.FileWriter('./graphs',sess.graph)
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

        coord.request_stop()
        coord.join(threads)

        writer.close()





main()
