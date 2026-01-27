from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import onnxruntime
import base64

app = Flask(__name__)

# --- Inisialisasi Model dan Kelas (tidak berubah) ---
# Pastikan daftar nama kelas Anda lengkap di sini
CLASS_NAMES = [
    'bantu', 'bapak', 'ibu', 'kamu', 'maaf', 'makan', 'mau', 'rumah', 'saya', 'sehat', 'tidur', 'tolong'
]
try:
    session = onnxruntime.InferenceSession("best.onnx")
    model_inputs = session.get_inputs()
    input_shape = model_inputs[0].shape
    input_width, input_height = input_shape[2], input_shape[3]
except Exception as e:
    print(f"Error loading ONNX model: {e}")
    session = None

# --- Routing Halaman ---

# Route untuk menyajikan Landing Page
@app.route('/')
def index():
    return render_template('index.html')

# Route untuk menyajikan Halaman Deteksi
@app.route('/deteksi')
def deteksi_page():
    return render_template('deteksi.html')

# --- API Endpoint untuk deteksi (tidak berubah) ---
@app.route('/detect', methods=['POST'])
def detect():
    if not session:
        return jsonify({"error": "Model not loaded"}), 500
    
    data = request.json
    image_data = base64.b64decode(data['image'].split(',')[1])
    np_arr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    img_height, img_width, _ = img.shape
    input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_img = cv2.resize(input_img, (input_width, input_height))
    input_img = input_img / 255.0
    input_img = input_img.transpose(2, 0, 1)
    input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
    outputs = session.run(None, {model_inputs[0].name: input_tensor})
    detections = []
    output_data = outputs[0][0].T
    for row in output_data:
        box = row[:4]
        score = row[4:].max()
        class_id = row[4:].argmax()
        if score > 0.5: # Ambang batas kepercayaan
            x, y, w, h = box
            x1 = int((x - w / 2) * img_width / input_width)
            y1 = int((y - h / 2) * img_height / input_height)
            x2 = int((x + w / 2) * img_width / input_width)
            y2 = int((y + h / 2) * img_height / input_height)
            detections.append({
                "box": [x1, y1, x2, y2],
                "class_name": CLASS_NAMES[int(class_id)],
                "score": float(score)
            })
    return jsonify(detections)

if __name__ == '__main__':
    app.run(debug=True)
