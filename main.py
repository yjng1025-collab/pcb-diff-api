from flask import Flask, request, jsonify, send_from_directory
import os
import base64
import cv2
import numpy as np
import requests
from io import BytesIO
from skimage.metrics import structural_similarity as ssim
from werkzeug.utils import secure_filename

app = Flask(__name__)

# 文件夹配置
UPLOAD_FOLDER = 'uploads'
STANDARD_FOLDER = 'standard_boards'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 创建必要目录
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STANDARD_FOLDER, exist_ok=True)
os.makedirs("static", exist_ok=True)

# 允许文件类型检查
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 图像比较函数
def compare_images(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        return None, None, None

    # 图像尺寸匹配
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    score, diff = ssim(gray1, gray2, full=True)
    diff = (diff * 255).astype("uint8")

    # 找出差异区域
    thresh = cv2.threshold(diff, 200, 255, cv2.THRESH_BINARY_INV)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = img2.copy()
    diff_coords = []

    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        diff_coords.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # 转成 base64
    _, buffer = cv2.imencode('.jpg', output)
    encoded_output = base64.b64encode(buffer).decode('utf-8')

    return score, encoded_output, diff_coords

@app.route('/')
def index():
    return "PCB Compare Tool is running."

@app.route('/compare_auto', methods=['POST'])
def compare_auto():
    # Step 1: 获取学生上传图像
    upload_path = None

    if 'image' in request.files:
        # 方式一：multipart/form-data 上传图片
        image = request.files['image']
        if image.filename == '' or not allowed_file(image.filename):
            return jsonify({'error': 'Invalid file'}), 400
        filename = secure_filename(image.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(upload_path)

    else:
        # 方式二：application/json 提供远程图片 URL
        data = request.get_json(silent=True)
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        image_url = data['image']
        try:
            response = requests.get(image_url)
            if response.status_code != 200:
                return jsonify({'error': 'Failed to download image'}), 400

            filename = 'downloaded_image.jpg'
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with open(upload_path, 'wb') as f:
                f.write(response.content)

        except Exception as e:
            return jsonify({'error': f'Error downloading image: {str(e)}'}), 500

    if not upload_path or not os.path.exists(upload_path):
        return jsonify({'error': 'Image processing failed'}), 500

    # Step 2: 与所有标准板图片比对
    best_score = -1
    best_result = None

    for std_file in os.listdir(STANDARD_FOLDER):
        if not allowed_file(std_file):
            continue

        std_path = os.path.join(STANDARD_FOLDER, std_file)
        score, output_image, diff_coords = compare_images(std_path, upload_path)

        if score is None:
            continue

        if score > best_score:
            best_score = score
            best_result = {
                "similarity": round(score, 4),
                "diff_coordinates": diff_coords,
                "result_image_base64": output_image,
                "standard_used": std_file
            }

    if not best_result:
        return jsonify({'error': 'No valid standard image found for comparison'}), 500

    return jsonify(best_result)

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)



