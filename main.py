from flask import Flask, request, jsonify, send_from_directory
import os
import base64
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from werkzeug.utils import secure_filename
from pathlib import Path

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
STANDARD_FOLDER = 'standard_boards'
STATIC_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/compare_auto', methods=['POST'])
def compare_auto():
    if 'student_image' not in request.files:
        return jsonify({"error": "Missing student_image"}), 400

    student_file = request.files['student_image']
    if not allowed_file(student_file.filename):
        return jsonify({"error": "Unsupported file format"}), 400

    student_img = cv2.imdecode(np.frombuffer(student_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Load first image in standard_board folder
    standard_files = os.listdir(STANDARD_FOLDER)
    if not standard_files:
        return jsonify({"error": "No standard images found"}), 500

    standard_path = os.path.join(STANDARD_FOLDER, standard_files[0])
    standard_img = cv2.imread(standard_path)

    # Resize to match
    student_img = cv2.resize(student_img, (standard_img.shape[1], standard_img.shape[0]))

    # Convert to grayscale
    student_gray = cv2.cvtColor(student_img, cv2.COLOR_BGR2GRAY)
    standard_gray = cv2.cvtColor(standard_img, cv2.COLOR_BGR2GRAY)

    # Compute SSIM
    similarity, diff = ssim(student_gray, standard_gray, full=True)
    diff = (diff * 255).astype("uint8")

    # Threshold to highlight differences
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw red boxes on differences
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(student_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Save result image to /static folder
    result_filename = "diff_result.jpg"
    result_path = os.path.join(STATIC_FOLDER, result_filename)
    cv2.imwrite(result_path, student_img)

    # Replace with your domain when deployed
    base_url = request.host_url.rstrip('/')
    image_url = f"{base_url}/static/{result_filename}"

    return jsonify({
        "similarity": round(similarity, 3),
        "diff_image_url": image_url
    })

# Optional: route to serve static files (Flask auto-handles /static if static_folder is set)
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(STATIC_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
