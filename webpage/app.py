from flask import Flask
from flask import render_template
from flask import request, jsonify
import cv2
import numpy as np
import base64
from PIL import Image
import io, sys
sys.path.insert(1, r'C:\Users\erik\OneDrive\Cross-Code\VSCode\PythonCode\Starting_Code\Machine Learning\pytorchlearning\Speed')
from getmodeloutput import Clicker

app = Flask(__name__)

judge = Clicker()

def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))

def toRGB(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

@app.route('/')
def hello_world():
    return render_template("main.html")

@app.route('/receive', methods=['POST'])
def get_post_json():    
    frames = []
    data = request.get_json()
    print(len(data))
    for imgdata in data:
        img = stringToImage(imgdata.split(",")[1])
        frames.append(toRGB(img))

    score = judge.getScore(frames)
    return jsonify(score)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, ssl_context='adhoc', threaded=True)