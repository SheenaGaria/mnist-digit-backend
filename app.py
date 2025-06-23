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
for_in range(5):
      dummy_input = np.random.rand(1, 1, 28, 28).astype(np.float32)
        input_name = session.get_inputs()[0].name
        _ = session.run(None, {input_name: dummy_input})

        # Create dummy image
        image_array = (dummy_input[0][0] * 255).astype(np.uint8)
        img = Image.fromarray(image_array)
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        images.append(img_b64)

    return jsonify({"images": images})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)