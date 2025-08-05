from flask import Flask, request, jsonify, send_from_directory
import os
import base64
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
STANDARD_FOLDER = 'standard_boards'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 创建必要的目录
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STANDARD_FOLDER, exist_ok=True)
os.makedirs("static", exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def compare_images(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, diff = ssim(gray1, gray2, full=True)
    diff = (diff * 255).astype("uint8")

    thresh = cv2.threshold(diff, 200, 255, cv2.THRESH_BINARY_INV)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = img2.copy()
    diff_coords = []

    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        diff_coords.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)

    _, buffer = cv2.imencode('.jpg', output)
    encoded_output = base64.b64encode(buffer).decode('utf-8')
    return score, encoded_output, diff_coords

@app.route('/')
def index():
    return "PCB Compare Tool is running."

@app.route('/compare_auto', methods=['POST'])
def compare_auto():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    if image.filename == '' or not allowed_file(image.filename):
        return jsonify({'error': 'Invalid file'}), 400

    filename = secure_filename(image.filename)
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(upload_path)

    best_score = -1
    best_result = None

    for std_file in os.listdir(STANDARD_FOLDER):
        if not allowed_file(std_file):
            continue

        std_path = os.path.join(STANDARD_FOLDER, std_file)
        score, output_image, diff_coords = compare_images(std_path, upload_path)

        if score > best_score:
            best_score = score
            best_result = {
                "similarity": score,
                "diff_coordinates": diff_coords,
                "result_image_base64": output_image,
                "standard_used": std_file
            }

    return jsonify(best_result)

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)

