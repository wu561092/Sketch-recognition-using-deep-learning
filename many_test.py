import tensorflow as tf
import numpy as np
import os
import glob
from PIL import Image

sketch_image_list=[]
sketch_label_list=[]
neg_image_list=[]
neg_label_list=[]

def read_sketch():
    s_txt=open('sketch_test.txt','r')
    for line in s_txt:
        line2=line.split()
        #print line2[0]
        sketch_image_list.append(line2[0])
        sketch_label_list.append(int(line2[1]))



def read_postive():
    n_txt=open('pos_test.txt','r')
    for line in n_txt:
        line2=line.split()
        neg_image_list.append(line2[0])
        neg_label_list.append(int(line2[1]))
    #return neg_rgb,neg_label



with tf.gfile.FastGFile('y2_y3_p_softmax.pb','rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
 

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name("Lc_loss/p_softmax:0")
    read_sketch()
    read_postive()
    #for i in sketch_image_list and j in neg_image_list and k in sketch_label_list: 
    for i,j,k in zip (sketch_image_list,neg_image_list,sketch_label_list):
        sketch_img = Image.open(i)
        sketch_img = sketch_img.resize((227,227))
        sketch_rgbimg = sketch_img.convert('RGB')
        sketch_image=np.reshape(sketch_rgbimg, (1, 227,227,3))

        neg_img = Image.open(j)
        neg_img = neg_img.resize((227,227))
        neg_rgbimg = neg_img.convert('RGB')
        neg_image=np.reshape(neg_rgbimg, (1, 227,227,3))
        predictions = sess.run(softmax_tensor, {'sketch_input/sketch_input:0':sketch_image,'positive_input/positive_real_input:0':neg_image})  
        predictions = np.squeeze(predictions) 
        top_k = predictions.argsort()[::-1]
        print(top_k)
        if not(top_k[0]==k):
            print "wtf"
