import os
import tensorflow as tf
import numpy as np


def gci(filepath):
    # 遍历filepath下所有文件，包括子目录
    files = os.listdir(filepath)
    for fi in files:
        fi_d = os.path.join(filepath, fi)
        if os.path.isdir(fi_d):
            gci(fi_d)
        else:
            print(os.path.join(filepath, fi_d))


class Recognizer:
    def __init__(self):
        # workpath = os.getcwd()
        # workpath = sys.path[0]os.path.split(os.path.realpath(__file__))[0]
        workpath = os.path.split(os.path.realpath(__file__))[0]

        self.modelpath = os.path.join(workpath, r'Net')
        self.sess = tf.Session()
        # model_file = tf.train.latest_checkpoint(self.modelpath)
        model_file = os.path.join(self.modelpath, 'predictmodel.ckpt-20000')
        saver = tf.train.import_meta_graph(model_file + '.meta')
        # predictmodel.ckpt-20000
        saver.restore(self.sess, model_file)
        graph = tf.get_default_graph()
        self.x = graph.get_tensor_by_name("x:0")
        self.y = graph.get_tensor_by_name("y:0")
        self.keep = graph.get_tensor_by_name("keep:0")

    def rec(self, images, paramkeep=0.5):
        result = self.sess.run(self.y, feed_dict={self.x: images, self.keep: paramkeep})
        return np.argmax(result, axis=1), result
