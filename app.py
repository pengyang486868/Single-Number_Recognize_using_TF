# coding: utf-8
from flask import Flask, jsonify, request
import uuid
import json
import numpy as np
import os
from numrec import Recognizer
from PIL import Image

app = Flask(__name__)
r = Recognizer()
path = os.path.join(os.getcwd(), r'Pics')


# http://localhost:17088/test?path=single3
@app.route('/test')
def test():
    httpargs = request.args
    im = Image.open(os.path.join(path, httpargs['path'] + '.jpg'))
    imraw = [np.array(im)[:, :, 0].flatten()]
    return jsonify(str(r.rec(imraw)))


@app.route('/single', methods=['GET', 'POST'])
def ocr():
    pass



# 启动程序
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=17088)
