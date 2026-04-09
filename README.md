# 🤟 YOLOv11 for Sign Language Recognition (BISINDO & SIBI)

This project is an implementation of the YOLOv11 model to detect and recognize words in Indonesian Sign Language, specifically BISINDO (Bahasa Isyarat Indonesia) and SIBI (Sistem Isyarat Bahasa Indonesia).

The system is trained to identify hand gestures representing specific words such as *aku*, *kamu*, *makan*, and others, based on image data collected from deaf individuals and special education teachers (SLB).

---

## 🎯 Objective

The main objective of this project is to evaluate the performance of the YOLOv11 model in detecting and recognizing sign language gestures, as well as to support communication accessibility for the deaf community.

---

## 🧠 Model & Method

- Model: YOLOv11 (You Only Look Once)
- Task: Object Detection (Sign Language Words)
- Framework: PyTorch (Ultralytics YOLO)
- Data Annotation and Management: Roboflow
- Training Environment: Google Colab

---

## 📊 Dataset

- Source: Custom dataset
- Collected from:
  - Deaf individuals
  - public Datasets 
  - Special education teachers (SLB)
- Type: Image dataset
- Classes: Words in BISINDO & SIBI (e.g., *aku*, *kamu*, *makan*, etc.)

---

## ⚙️ Features

- Real-time sign language detection
- Word-based gesture recognition
- Custom-trained YOLOv11 model (`best.onnx`)
- Potential for deployment using Flask (web-based application)

---

## 🚀 How to Run

1. Clone this repository: git clone https://github.com/beginnoob/Yolov11-Bahasa-Isyarat.git
2. Install dependencies: pip install -r requirements.txt
3. Run inference: python app.py
