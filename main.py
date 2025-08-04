from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import base64

app = Flask(__name__)
CORS(app)  # ✅ 启用跨域请求支持

def compare_images(img1, img2):
    # 将 img2 调整为与 img1 相同大小
    if img1.shape[:2] != img2.shape[:2]:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    grayA = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, diff = ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    return score, diff

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

    score, diff = compare_images(img1, img2)
    _, buffer = cv2.imencode('.jpg', diff)
    encoded_diff = base64.b64encode(buffer).decode()

    return jsonify({
        "score": float(score),
        "diff_image": encoded_diff
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
