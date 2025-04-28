from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import cv2
import numpy as np
import piexif
import requests
from PIL import Image
from rembg import remove
import base64
import io

UPLOAD_FOLDER = "static"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# フォルダ作成
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ルート
@app.route("/", methods=["GET"])
def upload_file():
    return render_template("upload.html")

# アップロード→プレビュー表示
@app.route("/upload", methods=["POST"])
def handle_upload():
    file = request.files["image"]
    filename = file.filename
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # フォーム入力
    bgcolor = request.form.get("bgcolor", "255,255,255")
    aspect_ratio = request.form.get("aspect_ratio", "3:4")
    custom_rw = request.form.get("custom_rw_hidden", "3")
    custom_rh = request.form.get("custom_rh_hidden", "4")
    purpose = request.form.get("purpose", "job")
    width = int(request.form.get("width", 600))
    height = int(request.form.get("height", 800))
    format = request.form.get("format", "jpeg")
    scale_factor = float(request.form.get("scale_factor", 1))
    y_offset = int(request.form.get("y_offset", 0))

    # 背景除去
    input_image = Image.open(filepath)
    no_bg_image = remove(input_image)

    # 背景合成
    r, g, b = map(int, bgcolor.split(","))
    bg_color = (r, g, b)
    output_array = np.array(no_bg_image)
    if output_array.shape[-1] == 4:
        alpha = output_array[:, :, 3] / 255.0
        for c in range(3):
            output_array[:, :, c] = output_array[:, :, c] * alpha + bg_color[c] * (1 - alpha)
        output_array = output_array[:, :, :3]

    # サイズ・比率調整
    result_img = Image.fromarray(output_array.astype(np.uint8))

    iw, ih = result_img.size
    iw = int(iw * scale_factor)
    ih = int(ih * scale_factor)
    resized_img = result_img.resize((iw, ih))

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    x = (width - iw) // 2
    y = (height - ih) // 2 + y_offset
    canvas.paste(resized_img, (x, y))

    # プレビュー用保存
    preview_path = os.path.join(app.config["UPLOAD_FOLDER"], "preview.jpg")
    canvas.save(preview_path, format.upper())

    # base64エンコードしてHTMLへ
    with open(preview_path, "rb") as f:
        preview_b64 = base64.b64encode(f.read()).decode("utf-8")

    return render_template("preview.html", preview_image=preview_b64, filename=filename, bgcolor=bgcolor, 
                           aspect_ratio=aspect_ratio, custom_rw=custom_rw, custom_rh=custom_rh, 
                           purpose=purpose, width=width, height=height, format=format,
                           scale_factor=scale_factor, y_offset=y_offset)

# プレビュー後の処理（決済ページへ）
@app.route("/confirm", methods=["POST"])
def confirm():
    return redirect(url_for("success"))

# 決済成功後、本番用画像を生成
@app.route("/success", methods=["GET", "POST"])
def success():
    if request.method == "GET":
        return render_template("success.html", result_image="result.jpg")

    filename = request.form.get("filename")
    bgcolor = request.form.get("bgcolor")
    aspect_ratio = request.form.get("aspect_ratio")
    custom_rw = request.form.get("custom_rw")
    custom_rh = request.form.get("custom_rh")
    purpose = request.form.get("purpose")
    width = int(request.form.get("width"))
    height = int(request.form.get("height"))
    format = request.form.get("format")
    scale_factor = float(request.form.get("scale_factor"))
    y_offset = int(request.form.get("y_offset"))

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    input_image = Image.open(filepath)

    # remove.bg API呼び出し
    api_key = os.getenv("REMOVEBG_API_KEY")
    response = requests.post(
        "https://api.remove.bg/v1.0/removebg",
        files={"image_file": open(filepath, "rb")},
        data={"size": "auto"},
        headers={"X-Api-Key": api_key}
    )

    if response.status_code == requests.codes.ok:
        final_image = Image.open(io.BytesIO(response.content))
    else:
        return "Background removal failed: {}".format(response.text)

    # 背景合成
    r, g, b = map(int, bgcolor.split(","))
    bg_color = (r, g, b)
    final_array = np.array(final_image)
    if final_array.shape[-1] == 4:
        alpha = final_array[:, :, 3] / 255.0
        for c in range(3):
            final_array[:, :, c] = final_array[:, :, c] * alpha + bg_color[c] * (1 - alpha)
        final_array = final_array[:, :, :3]

    result_img = Image.fromarray(final_array.astype(np.uint8))

    iw, ih = result_img.size
    iw = int(iw * scale_factor)
    ih = int(ih * scale_factor)
    resized_img = result_img.resize((iw, ih))

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    x = (width - iw) // 2
    y = (height - ih) // 2 + y_offset
    canvas.paste(resized_img, (x, y))

    result_path = os.path.join(app.config["UPLOAD_FOLDER"], "result.jpg")
    canvas.save(result_path, format.upper())

    return render_template("success.html", result_image="result.jpg")

# ダウンロード
@app.route("/download/<filename>")
def download(filename):
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    return send_file(filepath, as_attachment=True)

import traceback

@app.errorhandler(Exception)
def handle_exception(e):
    # すべての例外をキャッチして、ログに出す
    print("=== Exception Occurred ===")
    print(traceback.format_exc())
    return "Internal Server Error", 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # PORT 環境変数を読む
    app.run(host="0.0.0.0", port=port, debug=False)  # 0.0.0.0でバインド
