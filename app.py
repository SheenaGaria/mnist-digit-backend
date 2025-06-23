from flask import Flask, request, jsonify
from flask_cors import CORS
import onnxruntime as ort
import numpy as np
import base64
from PIL import Image
from io import BytesIO

app = Flask(__name__)
CORS(app) 

# Load the ONNX model
session = ort.InferenceSession("mnist-8.onnx", providers=["CPUExecutionProvider"])


@app.route("/generate", methods=["POST"])
def generate_images():
    data = request.get_json()
    digit = int(data.get("digit", -1))
    if digit < 0 or digit > 9:
        return jsonify({"error": "Digit must be between 0 and 9"}), 400


images = []
for _ in range(5):
       img_array = np.zeros((1, 1, 28, 28), dtype=np.float32)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: img_array})
    prediction = output[0]

    image = Image.fromarray((img_array[0][0] * 255).astype(np.uint8))
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    images.append(img_b64)
    return jsonify({"images": images})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)