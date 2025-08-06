import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from skimage.metrics import structural_similarity as ssim
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app)

STANDARD_BOARDS_DIR = "standard_boards"

def read_image_from_file(file_storage):
    in_memory_file = BytesIO()
    file_storage.save(in_memory_file)
    data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

def compare_images(imageA, imageB):
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    score, diff = ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")

    thresh = cv2.threshold(diff, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    contours, _ = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = imageB.copy()
    differences = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area > 100:  # filter out small changes
            differences.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Convert result to base64 image
    _, buffer = cv2.imencode('.jpg', result)
    result_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "ssim": score,
        "differences": differences,
        "result_image": result_base64
    }

@app.route('/')
def index():
    return jsonify({"message": "PCB Compare API Ready."})

@app.route('/compare', methods=['POST'])
def compare():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({"error": "Missing images"}), 400

    image1 = read_image_from_file(request.files['image1'])
    image2 = read_image_from_file(request.files['image2'])

    if image1 is None or image2 is None:
        return jsonify({"error": "Invalid image data"}), 400

    result = compare_images(image1, image2)
    return jsonify(result)

@app.route('/compare_auto', methods=['POST'])
def compare_auto():
    if 'image' not in request.files:
        return jsonify({"error": "Missing image"}), 400

    uploaded_image = read_image_from_file(request.files['image'])
    if uploaded_image is None:
        return jsonify({"error": "Invalid image data"}), 400

    board_names = sorted(os.listdir(STANDARD_BOARDS_DIR))
    if not board_names:
        return jsonify({"error": "No standard boards found"}), 500

    # Load the first standard image for auto compare
    standard_path = os.path.join(STANDARD_BOARDS_DIR, board_names[0])
    standard_image = cv2.imread(standard_path)

    if standard_image is None:
        return jsonify({"error": f"Failed to load standard board image: {standard_path}"}), 500

    result = compare_images(uploaded_image, standard_image)
    result["standard_board_used"] = board_names[0]

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)



