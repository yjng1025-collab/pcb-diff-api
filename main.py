from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from werkzeug.utils import secure_filename

# === Config ===
UPLOAD_FOLDER = 'uploads'
STANDARD_FOLDER = 'standard_boards'
STATIC_FOLDER = 'static'
TEMPLATE_FOLDER = 'template'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# === Flask App ===
app = Flask(__name__, static_folder=STATIC_FOLDER, template_folder=TEMPLATE_FOLDER)

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STANDARD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# === Utility ===
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# === ROUTES ===

# Home page (to prevent 404)
@app.route('/')
def home():
    return render_template('index.html')  # Or just: return 'API is running'

# Compare uploaded image with first standard image
@app.route('/compare_auto', methods=['POST'])
def compare_auto():
    if 'student_image' not in request.files:
        return jsonify({"error": "Missing student_image"}), 400

    student_file = request.files['student_image']
    if not allowed_file(student_file.filename):
        return jsonify({"error": "Unsupported file format"}), 400

    student_img = cv2.imdecode(np.frombuffer(student_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Load first image in standard_boards
    standard_files = [f for f in os.listdir(STANDARD_FOLDER) if allowed_file(f)]
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

    # Highlight differences
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(student_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Save result image to /static
    result_filename = "diff_result.jpg"
    result_path = os.path.join(STATIC_FOLDER, result_filename)
    cv2.imwrite(result_path, student_img)

    # Build public URL
    base_url = request.host_url.rstrip('/')
    image_url = f"{base_url}/static/{result_filename}"

    return jsonify({
        "similarity": round(similarity, 3),
        "diff_image_url": image_url
    })

# Optional route to serve static files manually (not needed if Flask auto-serves /static/)
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(STATIC_FOLDER, filename)

# === Main Entry ===
if __name__ == '__main__':
    app.run(debug=True)
