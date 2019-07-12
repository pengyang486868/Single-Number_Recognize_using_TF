import os
import tensorflow as tf
import numpy as np


class Recognizer:
    def __init__(self):
        # workpath = os.getcwd()
        # workpath = sys.path[0]os.path.split(os.path.realpath(__file__))[0]
        workpath = os.path.split(os.path.realpath(__file__))[0]
        self.modelpath = os.path.join(workpath, r'Net')
        self.sess = tf.Session()
        model_file = tf.train.latest_checkpoint(self.modelpath)
        saver = tf.train.import_meta_graph(model_file + '.meta')
        saver.restore(self.sess, model_file)
        graph = tf.get_default_graph()
        self.x = graph.get_tensor_by_name("x:0")
        self.y = graph.get_tensor_by_name("y:0")
        self.keep = graph.get_tensor_by_name("keep:0")

    def rec(self, images, paramkeep=0.5):
        result = self.sess.run(self.y, feed_dict={self.x: images, self.keep: paramkeep})
        return np.argmax(result, axis=1), result
