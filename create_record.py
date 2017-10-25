import tensorflow as tf
import numpy as np
import os
import glob
import tensorflow as tf
from PIL import Image
from tensorflow.python.framework.graph_util import convert_variables_to_constants


config=tf.ConfigProto()
config.gpu_options.allow_growth=True






NUM_CLASSES = 10 # each class train data number
WIDTH = 227
HEIGHT = 227





def s_create_record():
    s_txt=open('sketch.txt','r')
    writer = tf.python_io.TFRecordWriter("sketch_train.tfrecords")
    for line in s_txt:
        line2=line.split()
        img = Image.open(line2[0])
        num= int(line2[1])
        img = img.resize((WIDTH,HEIGHT))
        img_raw = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[num])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                 }))
        writer.write(example.SerializeToString())
    writer.close()
 

def p_create_record():
    p_txt=open('pos.txt','r')
    writer = tf.python_io.TFRecordWriter("pos_train.tfrecords")
    for line in p_txt:
        line2=line.split()
        img = Image.open(line2[0])
        num= int(line2[1])
        img = img.resize((WIDTH,HEIGHT))
        img_raw = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[num])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                 }))
        writer.write(example.SerializeToString())
    writer.close()

def n_create_record():
    n_txt=open('neg.txt','r')
    writer = tf.python_io.TFRecordWriter("neg_train3.tfrecords")
    for line in n_txt:
        line2=line.split()
        img = Image.open(line2[0])
        num= int(line2[1])
        img = img.resize((WIDTH,HEIGHT))
        print img
        rgbimg = img.convert('RGB')
        print rgbimg
        img_raw = rgbimg.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[num])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                 }))
        writer.write(example.SerializeToString())
    writer.close()

def main():
    
    s_create_record()
    p_create_record()
    n_create_record()

main()
