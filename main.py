from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import base64

app = Flask(__name__)
CORS(app)

def compare_images_and_mark(img1, img2, min_area=500):
    # 调整 img2 尺寸与 img1 一致
    if img1.shape[:2] != img2.shape[:2]:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    grayA = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, diff = ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")

    # 获取差异区域轮廓
    thresh = cv2.threshold(diff, 180, 255, cv2.THRESH_BINARY_INV)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 在学生图像上画出差异
    marked_img = img2.copy()
    for c in contours:
        area = cv2.contourArea(c)
        if area > min_area:  # 忽略太小的区域
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(marked_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return score, marked_img

@app.route("/", methods=["GET"])
def hello():
    return "PCB Image Compare API is running!"

@app.route("/compare", methods=["POST"])
def compare():
    if "img1" not in request.files or "img2" not in request.files:
        return jsonify({"error": "Please upload both img1 and img2"}), 400

    img1 = cv2.imdecode(np.frombuffer(request.files["img1"].read(), np.uint8), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np.frombuffer(request.files["img2"].read(), np.uint8), cv2.IMREAD_COLOR)

    if img1 is None or img2 is None:
        return jsonify({"error": "Failed to decode one or both images."}), 400

    score, marked_img = compare_images_and_mark(img1, img2)
    _, buffer = cv2.imencode('.jpg', marked_img)
    encoded_img = base64.b64encode(buffer).decode()

    return jsonify({
        "score": float(score),
        "marked_image": encoded_img
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
