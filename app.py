from flask import Flask, request, jsonify
from flask_cors import CORS
import onnxruntime as ort
import numpy as np
import base64
from PIL import Image
from io import BytesIO
import random

app = Flask(__name__)
CORS(app)  # Allow CORS for all domains (Vercel frontend access)

# Load the ONNX model
session = ort.InferenceSession("mnist-8.onnx", providers=["CPUExecutionProvider"])

# Load MNIST from local .npz file
data = np.load("mnist.npz")
x_train, y_train = data['x_train'], data['y_train']


# Load sample MNIST dataset images for generation
# You must have `mnist.npz` file generated using: 
# np.savez('mnist.npz', x_train=x_train, y_train=y_train)
data = np.load("mnist.npz")
x_train, y_train = data["x_train"], data["y_train"]

# Function to get 5 random images for a digit (0–9)
def get_images_for_digit(digit, count=5):
    indices = np.where(y_train == digit)[0]
    selected = np.random.choice(indices, count, replace=False)
    images = []

    for idx in selected:
        img_array = x_train[idx]

        # Preprocess image for ONNX model
        input_tensor = img_array.astype(np.float32) / 255.0
        input_tensor = input_tensor.reshape(1, 1, 28, 28)

        # Run inference
        outputs = session.run(None, {"Input3": input_tensor})
        prediction = np.argmax(outputs[0])

        # Convert image to base64 for web display
        img = Image.fromarray(img_array)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        images.append(img_b64)

    return images

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    digit = int(data.get("digit", -1))
    if digit < 0 or digit > 9:
        return jsonify({"error": "Digit must be between 0 and 9"}), 400

    try:
        images = get_images_for_digit(digit)
        return jsonify({"images": images})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
