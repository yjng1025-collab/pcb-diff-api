from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
STANDARD_FOLDER = 'standard_boards'
STATIC_FOLDER = 'static'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STANDARD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_similarity(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    img2_resized = cv2.resize(img2_gray, (img1_gray.shape[1], img1_gray.shape[0]))

    score, diff = ssim(img1_gray, img2_resized, full=True)
    diff = (diff * 255).astype("uint8")

    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img2_marked = cv2.resize(img2.copy(), (img1.shape[1], img1.shape[0]))
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img2_marked, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return score, img2_marked

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare_auto', methods=['POST'])
def compare_auto():
    if 'image' not in request.files:
        return jsonify({'error': 'Missing image'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected image'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        student_img = cv2.imread(filepath)
        best_score = -1
        best_img = None
        best_standard_name = None

        for std_name in os.listdir(STANDARD_FOLDER):
            std_path = os.path.join(STANDARD_FOLDER, std_name)
            if allowed_file(std_name):
                std_img = cv2.imread(std_path)
                try:
                    score, _ = calculate_similarity(student_img, std_img)
                except Exception:
                    continue
                if score > best_score:
                    best_score = score
                    best_img = std_img
                    best_standard_name = std_name

        if best_img is None:
            return jsonify({'error': 'No valid standard image found'}), 500

        score, diff_img = calculate_similarity(student_img, best_img)

        output_path = os.path.join(STATIC_FOLDER, 'diff_result.jpg')
        cv2.imwrite(output_path, diff_img)

        return jsonify({
            'similarity': round(score, 3),
            'diff_image_url': request.url_root + 'static/diff_result.jpg'
        })

    return jsonify({'error': 'Invalid file'}), 400

if __name__ == '__main__':
    app.run(debug=True)
