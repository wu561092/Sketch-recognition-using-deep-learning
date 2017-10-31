import tensorflow as tf
import numpy as np
import os
import glob
from PIL import Image

def read_sketch():
    n_txt=open('sketch_test.txt','r')
    for line in n_txt:
        line2=line.split()
        img = Image.open(line2[0])
        num= int(line2[1])
        img = img.resize((227,227))
        rgbimg = img.convert('RGB')
    return rgbimg
def read_postive():
    n_txt=open('pos_test.txt','r')
    for line in n_txt:
        line2=line.split()
        img = Image.open(line2[0])
        num= int(line2[1])
        img = img.resize((227,227))
        rgbimg = img.convert('RGB')
    return rgbimg



with tf.gfile.FastGFile('y2_y3_p_softmax.pb','rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
 

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name("Lc_loss/p_softmax:0")
    sketch_image=read_sketch()
    sketch_image=np.reshape(sketch_image, (1, 227,227,3))
    postive_image=read_postive()
    postive_image=np.reshape(postive_image, (1,227,227, 3))
    predictions = sess.run(softmax_tensor, {'sketch_input/sketch_input:0':sketch_image,'positive_input/positive_real_input:0':postive_image})  
    predictions = np.squeeze(predictions) 
 #   image_path = os.path.join(root, file)
    top_k = predictions.argsort()[::-1]
    print(top_k)
