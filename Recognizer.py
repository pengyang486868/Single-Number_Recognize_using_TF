import os
import tensorflow as tf
import numpy as np


class Recognizer:
    def __init__(self):
        workpath = os.getcwd()
        self.modelpath = os.path.join(workpath, r'Net')
        self.sess = tf.Session()
        model_file = tf.train.latest_checkpoint(self.modelpath)
        saver = tf.train.import_meta_graph(model_file + '.meta')
        saver.restore(self.sess, model_file)
        graph = tf.get_default_graph()
        self.x = graph.get_tensor_by_name("x:0")
        self.y = graph.get_tensor_by_name("y:0")
        self.keep = graph.get_tensor_by_name("keep:0")

    def rec(self, images):
        result_test = self.sess.run(self.y, feed_dict={self.x: images, self.keep: 0.5})
        return np.argmax(result_test, axis=1)
