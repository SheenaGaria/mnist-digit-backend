from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import random

app = Flask(__name__)
CORS(app)

def generate_digit_image(digit):
    return np.random.rand(28, 28).tolist()  # Simulates an MNIST-like image

@app.route("/generate-digit", methods=["POST"])
def generate_digit():
    data = request.get_json()
    digit = int(data.get("digit", 0))
    images = [generate_digit_image(digit) for _ in range(5)]
    return jsonify({"digit": digit, "images": images})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
