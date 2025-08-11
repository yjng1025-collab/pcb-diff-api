from flask import Flask, request, jsonify  
import os
import base64
import cv2
import numpy as np
import requests
from io import BytesIO
from skimage.metrics import structural_similarity as ssim
from datetime import datetime

app = Flask(__name__)

STANDARD_FOLDER = 'standard_boards'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# 确保 static 文件夹存在
os.makedirs("static", exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_image_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"Error loading image from URL: {e}")
        return None

def compare_images(img1, img2):
    MIN_AREA = 1200   # Ignore very small differences
    MAX_AREA_RATIO = 0.85  # Ignore if a blob covers >85% of image

    # Convert to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Resize to smallest common size
    height = min(img1_gray.shape[0], img2_gray.shape[0])
    width = min(img1_gray.shape[1], img2_gray.shape[1])
    img1_gray = cv2.resize(img1_gray, (width, height))
    img2_gray = cv2.resize(img2_gray, (width, height))

    # Compute SSIM
    score, diff = ssim(img1_gray, img2_gray, full=True)
    diff = (diff * 255).astype("uint8")

    # Blur to remove small variations (lighting, texture)
    diff_blur = cv2.GaussianBlur(diff, (5, 5), 0)

    # Threshold
    thresh = cv2.threshold(diff_blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result_img = cv2.resize(img2.copy(), (width, height))
    differences = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        if w > 10 and h > 10 and area >= MIN_AREA:
            # Skip if it's basically the entire image
            if area / (width * height) > MAX_AREA_RATIO:
                continue
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            differences.append({"x": int(x), "y": int(y), "width": int(w), "height": int(h)})

    return result_img, score, differences

@app.route("/compare_auto", methods=["POST"])
def compare_auto():
    data = request.get_json()

    if not data or "image" not in data:
        return jsonify({"error": "Missing image"}), 400

    student_img_url = data["image"]
    student_img = load_image_from_url(student_img_url)
    if student_img is None:
        return jsonify({"error": "Failed to load student image"}), 400

    best_score = -1
    best_result = None
    best_standard_name = None
    differences = []

    for filename in os.listdir(STANDARD_FOLDER):
        if allowed_file(filename):
            standard_path = os.path.join(STANDARD_FOLDER, filename)
            standard_img = cv2.imread(standard_path)
            if standard_img is None:
                continue

            result_img, score, diffs = compare_images(student_img, standard_img)
            if score > best_score:
                best_score = score
                best_result = result_img
                best_standard_name = filename
                differences = diffs

    if best_result is None:
        return jsonify({"error": "No valid standard images found"}), 500

    # ✅ 保存为唯一文件名（加时间戳）
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    output_filename = f"diff_{timestamp}.jpg"
    output_path = os.path.join("static", output_filename)
    cv2.imwrite(output_path, best_result)

    # ✅ 构造 URL
    server_url = request.host_url.rstrip("/")  # 去除末尾斜杠
    image_url = f"{server_url}/static/{output_filename}"

    # ✅ 生成差异描述（description）
    num_diffs = len(differences)
    if num_diffs == 0:
        description = "未发现明显差异"
    elif num_diffs < 5:
        description = f"发现 {num_diffs} 处轻微差异"
    elif num_diffs < 15:
        description = f"发现 {num_diffs} 处差异，分布较分散"
    else:
        description = f"发现 {num_diffs} 处差异，分布密集，可能存在较大问题"

    return jsonify({
        "matched_with": best_standard_name,
        "similarity": round(best_score, 4),
        "diff_image_url": image_url,
        "image": image_url,  # ✅ Coze-friendly 字段，自动渲染图片
        "differences": differences,
        "description": description  # ✅ 新增字段
    })

@app.route("/", methods=["GET"])
def root():
    return "<h2>PCB Diff API 正常运行</h2><p>使用 POST /compare_auto 上传学生图片 URL 并自动比对。</p>"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

