from flask import Flask, request
import requests
app = Flask(__name__)

@app.route('/', methods=["POST"])
def hello_world():
    print(request.json)
    targeturl = request.json
    r = requests.get(targeturl, allow_redirects=True)
    print(f"Saving video from {targeturl}")
    open('testvideo.mp4', 'wb').write(r.content)
    return "Hello World"

if __name__ == '__main__':
   app.run(port=8000)