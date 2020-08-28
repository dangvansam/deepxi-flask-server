import os
from flask import Flask, request, jsonify
from PIL import Image
app = Flask(__name__)

@app.route("/im_size", methods=["POST"])
def process_image():
    file = request.files['image']
    # Read the image via file.stream
    img = Image.open(file.stream)

    return jsonify({'msg': 'success', 'size': [img.width, img.height]})

@app.route("/audio", methods=["POST"])
def process_audio():
    file = request.files['audio']
    # Read the image via file.stream
    file.save(os.path.join('./upload/', file.filename))
    return jsonify({'msg': file.filename})


if __name__ == "__main__":
    app.run(host='192.168.1.254', port='9002',debug=True)