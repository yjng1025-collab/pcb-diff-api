import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
STANDARD_FOLDER = 'standard_boards'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STANDARD_FOLDER, exist_ok=True)


def compare_images(img1, img2):
    # Resize images to same size if needed
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    score, diff = ssim(gray1, gray2, full=True)
    diff = (diff * 255).astype("uint8")

    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotated = img2.copy()
    diff_boxes = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < 100:  # skip tiny noise
            continue
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
        diff_boxes.append({'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)})

    return annotated, score, diff_boxes


def encode_image(img):
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64


@app.route('/')
def home():
    return "PCB Compare API is running."


@app.route('/compare', methods=['POST'])
def compare():
    if 'student' not in request.files or 'standard' not in request.files:
        return jsonify({'error': 'Missing images'}), 400

    student_file = request.files['student']
    standard_file = request.files['standard']

    student_img = cv2.imdecode(np.frombuffer(student_file.read(), np.uint8), cv2.IMREAD_COLOR)
    standard_img = cv2.imdecode(np.frombuffer(standard_file.read(), np.uint8), cv2.IMREAD_COLOR)

    result_img, score, boxes = compare_images(standard_img, student_img)
    encoded_result = encode_image(result_img)

    return jsonify({
        'ssim_score': round(score, 4),
        'diff_image': encoded_result,
        'difference_boxes': boxes
    })


@app.route('/compare_auto', methods=['POST'])
def compare_auto():
    if 'image' not in request.files:
        return jsonify({'error': 'Missing student image'}), 400

    student_file = request.files['image']
    student_img = cv2.imdecode(np.frombuffer(student_file.read(), np.uint8), cv2.IMREAD_COLOR)

    best_score = -1
    best_result = None
    best_filename = None

    for filename in os.listdir(STANDARD_FOLDER):
        if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        path = os.path.join(STANDARD_FOLDER, filename)
        standard_img = cv2.imread(path)

        try:
            result_img, score, boxes = compare_images(standard_img, student_img)
            if score > best_score:
                best_score = score
                best_result = {
                    'ssim_score': round(score, 4),
                    'diff_image': encode_image(result_img),
                    'difference_boxes': boxes,
                    'matched_standard': filename
                }
        except Exception as e:
            continue  # Skip images that fail to compare

    if best_result is None:
        return jsonify({'error': 'No valid standard images found.'}), 500

    return jsonify(best_result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)


