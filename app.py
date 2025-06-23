from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.datasets import mnist
import base64
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# Load MNIST dataset once
(x_train, y_train), _ = mnist.load_data()

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

@app.route('/generate', methods=['POST'])
def generate_images():
    data = request.get_json()
    digit = int(data.get("digit", -1))
    if digit < 0 or digit > 9:
        return jsonify({"error": "Digit must be between 0 and 9"}), 400

    images = get_images_for_digit(digit)
    return jsonify({"images": images})

if __name__ == '__main__':
    app.run(debug=True)
