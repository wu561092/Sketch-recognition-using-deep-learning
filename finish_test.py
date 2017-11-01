import tensorflow as tf
import numpy as np
import os
import glob
from PIL import Image
from operator import itemgetter


path="/home/applyACC/persons/wu561092/8_22/test/"

yes_num=0


def read_sketch(i):
    sketch=path+"sketch_test"+str(i)+".txt"
    s_txt=open(sketch,'r')
    sketch_image_list=[]
    sketch_label_list=[]
    for line in s_txt:
        line2=line.split()
        sketch_image_list.append(line2[0])
        sketch_label_list.append(int(line2[1]))
    return sketch_image_list,sketch_label_list


def read_negtive(i):
    neg=path+"neg_test"+str(i)+".txt"
    n_txt=open(neg,'r')
    neg_image_list=[]
    neg_label_list=[]
    for line in n_txt:
        line2=line.split()
        neg_image_list.append(line2[0])
        neg_label_list.append(int(line2[1]))
    return neg_image_list,neg_label_list


with tf.gfile.FastGFile('p_soft_model_7810.pb','rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name("Lc_loss/p_softmax:0")
    for l in range(1,101):
        sketch_image_list,sketch_label_list=read_sketch(l)
        neg_image_list,neg_label_list=read_negtive(l)
        cnt=[0,0,0,0,0,0,0,0,0,0]
        for i,j,k in zip (sketch_image_list,neg_image_list,sketch_label_list):
            sketch_img = Image.open(i)
            sketch_img = sketch_img.resize((227,227))
            sketch_rgbimg = sketch_img.convert('RGB')
            sketch_image=np.reshape(sketch_rgbimg, (1, 227,227,3))

            neg_img = Image.open(j)
            neg_img = neg_img.resize((227,227))
            neg_rgbimg = neg_img.convert('RGB')
            neg_image=np.reshape(neg_rgbimg, (1, 227,227,3))
            predictions = sess.run(softmax_tensor, {'sketch_input/sketch_input:0':sketch_image,'nagative_input/negative_real_input:0':neg_image})
            predictions = np.squeeze(predictions)
            top_k = predictions.argsort()[::-1]
            cnt[top_k[0]]=cnt[top_k[0]]+1
            cate=0
        all_list=[]
        for m in cnt:
            print "mm",m
            allcnt=(m,cate)
            all_list.append(allcnt)
            cate=cate+1
        all_list.sort(key=itemgetter(0),reverse=True)
        for m,cate in all_list[:-5]:
            print "pair",m,cate
            if cate==k:
                print "mk",cate
                yes_num=yes_num+1
print yes_num
