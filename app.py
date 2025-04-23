# app.py
from flask import Flask, render_template, request, send_from_directory, redirect, url_for
import os
import cv2
import numpy as np
from rembg import remove
from PIL import Image
import io
from dotenv import load_dotenv
import stripe

load_dotenv()

app = Flask(__name__)
UPLOAD_FOLDER = "static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
YOUR_DOMAIN = "https://photo-bg-remover.onrender.com"

def preview_image(input_path, output_path, bgcolor=(255, 255, 255), threshold=0.5):
    with open(input_path, 'rb') as i:
        input_bytes = i.read()
    output_bytes = remove(input_bytes)

    image_pil = Image.open(io.BytesIO(output_bytes)).convert("RGBA")
    image = np.array(image_pil)
    h, w = image.shape[:2]

    alpha = image[:, :, 3].astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    eroded_alpha = cv2.erode(alpha, kernel, iterations=1)
    alpha_norm = eroded_alpha.astype(np.float32) / 255.0
    alpha_norm[alpha_norm < threshold] = 0.0

    foreground_rgb = image[:, :, :3].astype(np.float32)
    background = np.full((h, w, 3), bgcolor, dtype=np.float32)

    blended_rgb = foreground_rgb * alpha_norm[:, :, None] + background * (1 - alpha_norm[:, :, None])
    blended_rgb = np.clip(blended_rgb, 0, 255).astype(np.uint8)
    blended_bgr = cv2.cvtColor(blended_rgb, cv2.COLOR_RGB2BGR)

    overlay = blended_bgr.copy()
    text = "SAMPLE"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = w / 400
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (w - text_size[0]) // 2
    text_y = (h + text_size[1]) // 2
    cv2.putText(overlay, text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

    result = cv2.addWeighted(overlay, 0.5, blended_bgr, 0.5, 0)
    cv2.imwrite(output_path, result)

def final_image(input_path, output_path, bgcolor=(255, 255, 255), size=(600, 800), fmt="png", y_offset=0, scale_factor=1.0):
    with open(input_path, 'rb') as i:
        input_bytes = i.read()
    output_bytes = remove(input_bytes)

    image_pil = Image.open(io.BytesIO(output_bytes)).convert("RGBA")
    image = np.array(image_pil)
    h, w = image.shape[:2]

    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((h, w, 4), dtype=np.uint8)
    canvas[:, :, :] = (255, 255, 255, 0)
    start_y = (h - new_h) // 2 + int(y_offset)
    start_x = (w - new_w) // 2
    if 0 <= start_y < h and 0 <= start_x < w:
        y1 = max(0, start_y)
        y2 = min(h, start_y + new_h)
        x1 = max(0, start_x)
        x2 = min(w, start_x + new_w)
        canvas[y1:y2, x1:x2] = image[0:(y2 - y1), 0:(x2 - x1)]
    else:
        canvas = cv2.resize(image, (w, h))

    alpha = canvas[:, :, 3].astype(np.float32) / 255.0
    foreground = canvas[:, :, :3].astype(np.float32)
    background = np.full((h, w, 3), bgcolor, dtype=np.float32)

    blended = foreground * alpha[..., None] + background * (1 - alpha[..., None])
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    result_img = Image.fromarray(blended.astype(np.uint8)).convert("RGB")
    result_img = result_img.resize(size)
    result_img.save(output_path, format=fmt.upper())

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
    image = request.files["image"]
    threshold = float(request.form.get("alpha_threshold", 0.5))
    bgcolor_str = request.form.get("bgcolor", "255,255,255")
    bgcolor = tuple(map(int, bgcolor_str.split(",")))
    width = int(request.form.get("width", 600))
    height = int(request.form.get("height", 800))
    aspect_ratio = request.form.get("aspect_ratio", "3:4")
    custom_rw = request.form.get("custom_rw", "3")
    custom_rh = request.form.get("custom_rh", "4")
    purpose = request.form.get("purpose", "job")
    fmt = request.form.get("format", "png")

    filename = "uploaded_image.jpg"
    input_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    image.save(input_path)

    preview_path = os.path.join(app.config["UPLOAD_FOLDER"], "preview.jpg")
    preview_image(input_path, preview_path, bgcolor, threshold)

    return render_template("preview.html", filename="preview.jpg", bgcolor=bgcolor_str, width=width, height=height, format=fmt, purpose=purpose, aspect_ratio=aspect_ratio, custom_rw=custom_rw, custom_rh=custom_rh)

@app.route("/create-checkout-session", methods=["POST"])
def create_checkout_session():
    bgcolor = request.form.get("bgcolor", "255,255,255")
    width = request.form.get("width", "600")
    height = request.form.get("height", "800")
    fmt = request.form.get("format", "png")
    y_offset = request.form.get("y_offset", "0")
    scale = request.form.get("scale_factor", "1.0")
    purpose = request.form.get("purpose", "job")
    aspect_ratio = request.form.get("aspect_ratio", "3:4")
    custom_rw = request.form.get("custom_rw", "3")
    custom_rh = request.form.get("custom_rh", "4")
    session = stripe.checkout.Session.create(
        payment_method_types=["card"],
        line_items=[{
            "price_data": {
                "currency": "jpy",
                "product_data": {"name": "証明写真（背景透過・高品質）"},
                "unit_amount": 300,
            },
            "quantity": 1,
        }],
        mode="payment",
        success_url=YOUR_DOMAIN + f"/success?bgcolor={bgcolor}&width={width}&height={height}&format={fmt}&y_offset={y_offset}&scale={scale}&purpose={purpose}&aspect_ratio={aspect_ratio}&custom_rw={custom_rw}&custom_rh={custom_rh}",
        cancel_url=YOUR_DOMAIN + "/cancel",
    )
    return redirect(session.url, code=303)

@app.route("/success")
def success():
    input_path = os.path.join(app.config["UPLOAD_FOLDER"], "uploaded_image.jpg")
    final_path = os.path.join(app.config["UPLOAD_FOLDER"], "final.png")

    bgcolor_str = request.args.get("bgcolor", "255,255,255")
    bgcolor = tuple(map(int, bgcolor_str.split(",")))
    width = int(request.args.get("width", 600))
    height = int(request.args.get("height", 800))
    fmt = request.args.get("format", "png")
    y_offset = int(request.args.get("y_offset", 0))
    scale = float(request.args.get("scale", 1.0))

    purpose = request.args.get("purpose", "job")
    aspect_ratio = request.args.get("aspect_ratio", "3:4")
    custom_rw = request.args.get("custom_rw", "3")
    custom_rh = request.args.get("custom_rh", "4")

    final_image(input_path, final_path, bgcolor, (width, height), fmt, y_offset, scale)

    return render_template("success.html",
                           filename="final.png",
                           purpose=purpose,
                           aspect_ratio=aspect_ratio,
                           custom_rw=custom_rw,
                           custom_rh=custom_rh)

@app.route("/download")
def download():
    return send_from_directory(app.config["UPLOAD_FOLDER"], "final.png", as_attachment=True)

@app.route("/cancel")
def cancel():
    return "<h1>決済がキャンセルされました。</h1>"

@app.route("/static/<filename>")
def send_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)


