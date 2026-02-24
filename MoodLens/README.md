# 🎭 MoodLens – Face Recognition & Emotion Analysis System

MoodLens is a computer vision project that performs real-time face detection and emotion analysis on video streams.

The system processes video frame-by-frame and predicts the dominant emotional state of detected individuals.

---

## 📌 Project Overview

MoodLens can:

- Detect faces from webcam or video files
- Analyze emotional state
- Multi-person tracking
- Perform second-by-second emotion detection
- Display dominant emotion with confidence score

This project focuses on real-time AI-based human emotion analysis using deep learning techniques.

---

## 🛠️ Technologies Used

- Python
- OpenCV
- DeepFace
- MediaPipe
- NumPy

---

## 🧠 System Workflow

1. Video input is captured (webcam or file).
2. Each frame is processed.
3. Face detection is applied.
4. Detected faces are analyzed using DeepFace.
5. The dominant emotion and confidence score are calculated.
6. Results are displayed on the video frame.

---

## ▶️ Usage

Run with webcam:

python src/app.py

---

## 📊 Output

- Face detected
- Dominant emotion (Happy, Sad, Angry, etc.)
- Confidence score
- Real-time video display

---

## 🎯 Future Improvements

- Emotion timeline visualization
- Performance optimization
