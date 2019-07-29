# coding: utf-8
from flask import Flask, jsonify, request
import json
import numpy as np
import os
from numrec import Recognizer
from PIL import Image

app = Flask(__name__)
r = Recognizer()
path = os.path.join(os.getcwd(), r'Pics')


# http://localhost:17088/testserver?in=kwejqhih
@app.route('/testserver')
def testserver():
    httpargs = request.args
    return jsonify({'result': True, 'input': httpargs['in']})


# http://localhost:17088/test?path=single3
@app.route('/test')
def test():
    httpargs = request.args
    im = Image.open(os.path.join(path, httpargs['path'] + '.jpg'))
    imraw = [np.array(im)[:, :, 0].flatten()]
    result, info = r.rec(imraw, paramkeep=0.6)
    return jsonify({'result': str(result), 'probs': str(info)})


# http://localhost:17088/posttest
@app.route('/posttest', methods=['POST'])
def posttest():
    data = request.get_data()
    httpargs = json.loads(data.decode('utf-8'))
    result, info = r.rec(httpargs['img'], paramkeep=0.6)
    return jsonify({'result': str(result), 'probs': str(info)})


@app.route('/single', methods=['GET', 'POST'])
def ocr():
    if request.method == 'GET':
        return None
    if request.method == 'POST':
        data = request.get_data()
        httpargs = json.loads(data.decode('utf-8'))
        image_arr = httpargs['img']
        keep_param = 0.5
        if 'keep' in httpargs:
            keep_param = float(httpargs['keep'])

        result, info = r.rec(image_arr, paramkeep=keep_param)
        # return jsonify({'result': str(result), 'probs': str(info)})
        return jsonify({'result': result.tolist(), 'probs': info.tolist()})
    return None


# run server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=17088)
