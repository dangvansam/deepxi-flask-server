<<<<<<< HEAD
# -- coding:utf-8 --
import os
import io
import requests
from flask import Flask, send_file, escape, request, render_template, jsonify, make_response, session,send_from_directory
#from utils import cvtToWavMono16, split
import random
from flask_cors import CORS
from infer import get_model_new
UPLOAD_DIRECTORY = "./upload"

app = Flask(__name__)
CORS(app)
infer = get_model_new()
 
=======
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

>>>>>>> 8562128801c20609f75775b37774ad133c45dda9
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
<<<<<<< HEAD
        #if  os.path.splitext(file_path)[1][1:].strip() not in ['wav']:
        #    out_file_path =u'"%s.wav"' %(file_path[:-4])
        #    print ('out_file_path:',out_file_path)
        #    str =u'ffmpeg -i "%s" -acodec pcm_u8 -ar 16000 -ac 1 "%s"' %(file_path,out_file_path)
        #    print (str)
        #    os.system(str)
        #    file_path =out_file_path

        print('saved file: {}'.format(file_path))
        res = make_response(jsonify({"file_path": file_path, "message": "Đã lưu file : {} lên server".format(filename)}))
=======
        print('saved file: {}'.format(file_path))
        res = make_response(jsonify({"file_path": file_path, "message": "Saved: {} to server".format(filename)}))
>>>>>>> 8562128801c20609f75775b37774ad133c45dda9
        return res
    return render_template('index.html')

@app.route('/predict/<file_path>')
def predict(file_path):
    print(file_path)
    file_path = file_path.replace('=','/')
    out_file_path = infer(file_path)
    print('predict done!!')
<<<<<<< HEAD
    print('out_file_path:', out_file_path)
    res = make_response(jsonify({"out_file_path":out_file_path, "message": "Khử nhiễu hoàn tất!"}))
    return res

@app.route("/audio", methods=["POST"])
def process_audio():
    file = request.files['audio']
    file_path = os.path.join('./upload/', file.filename)
    file.save(file_path)
    out_file_path = infer(file_path)
    with open(out_file_path, 'rb') as bites:
        return send_file(io.BytesIO(bites.read()),attachment_filename='out.wav',mimetype='audio/wav')
@app.route("/audio_url", methods=["POST"])
def process_audio_url():
    #print ('request.files:', request.files)
    #print ('request:',request)
    #print ('request.data:',request.data)
    file = request.files['audio']
    userId =  request.args.get('userId') #request['userId']
    recordId = request.args.get('recordId') # request['recordId']
    print ('userId:', userId, ' recordId:', recordId)
    file_path = os.path.join('./upload/', file.filename)
    file.save(file_path)
    out_file_path = infer(file_path)
    return jsonify({'url': out_file_path, 'type': 'noise', 'userId': userId, 'recordId': recordId, 'status':'Hoàn thành'})

if __name__ == "__main__":
    app.debug = True
    app.secret_key = 'dangvansam'
    app.run(host='192.168.1.254', port='9002')
=======
    res = make_response(jsonify({"out_file_path":out_file_path, "message": "Predict susscess!"}))
    return res

if __name__ == "__main__":
    app.debug = True
    app.secret_key = 'dangvansam'
    #app.run(host='192.168.1.254', port='9002')
    app.run(port='8080')
>>>>>>> 8562128801c20609f75775b37774ad133c45dda9
