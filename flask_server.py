import os
import requests
from flask import Flask, escape, request, render_template, jsonify, make_response, session
#from utils import cvtToWavMono16, split

import random
from infer_file import get_model
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
infer = get_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST','GET'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        filename = file.filename
        print(filename)
        if  os.path.splitext(filename)[1][1:].strip() not in ['mp3','wav','flac']:
            return render_template('index.html', filename='{} file not support. select mp3, wav or flac!'.format(filename))      
        file_path = 'static/upload/' + filename
        file.save(file_path)
        print('saved file: {}'.format(file_path))
        res = make_response(jsonify({"file_path": file_path, "message": "Saved: {} to server".format(filename)}))
        return res
    return render_template('index.html')

@app.route('/predict/<file_path>')
def predict(file_path):
    print(file_path)
    file_path = file_path.replace('=','/')
    out_file_path = infer(file_path)
    print('predict done!!')
    res = make_response(jsonify({"out_file_path":out_file_path, "message": "Predict susscess!"}))
    return res

if __name__ == "__main__":
    app.debug = True
    app.secret_key = 'dangvansam'
    #app.run(host='192.168.1.254', port='9002')
    app.run(port='8080')