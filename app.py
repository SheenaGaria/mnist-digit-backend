from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import base64
from io import BytesIO

app = Flask(__name__)
CORS(app)

# Load MNIST dataset from saved .npz
data = np.load("mnist.npz")
x_train, y_train = data["x_train"], data["y_train"]

def get_images_for_digit(digit, count=5):
    indices = np.where(y_train == digit)[0]
    selected = np.random.choice(indices, count, replace=False)
    images = []

    for idx in selected:
        img = Image.fromarray(x_train[idx])
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

    images = get_images_for_digit(digit)
    return jsonify({"images": images})

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
