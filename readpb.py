import tensorflow as tf
from tensorflow.python.platform import gfile
with tf.gfile.FastGFile('./2fu_13000_paper_y2_y3wtf.pb','r') as f:
  graph_def=tf.GraphDef()
  graph_def.ParseFromString(f.read())
  tf.import_graph_def(graph_def,name='')
  for n in tf.get_default_graph().as_graph_def().node:
    print n.name
  
