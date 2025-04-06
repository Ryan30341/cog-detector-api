
from flask import Flask, request, jsonify
from cog_detector import cog_detector
import cv2
import numpy as np

app = Flask(__name__)
detector = cog_detector.CogDetector()

@app.route('/detect', methods=['POST'])
def detect_teeth():
    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    result = detector.detect(img)
    return jsonify({'teeth_count': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
