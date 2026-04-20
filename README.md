
# Sign Language to Text Converter using Computer Vision and NLP

This project implements a real-time Sign Language Recognition system that captures hand gestures via webcam, classifies them as ASL alphabets using a machine learning model, and refines the recognized text using **TextBlob** for grammatical correction. It is built with **OpenCV**, **MediaPipe**, **scikit-learn**, and **TextBlob**.

---

## 🧠 Project Overview

### Pipeline:

1. **Image Collection** - Capture gesture images for each class using a webcam.
2. **Dataset Creation** - Extract hand landmarks from collected images using MediaPipe.
3. **Model Training** - Train a Random Forest classifier on extracted features.
4. **Inference & Text Generation** - Run a real-time prediction system that:
   - Detects ASL alphabets
   - Forms words
   - Auto-corrects spelling using TextBlob
   - Displays final sentence on screen

---
## Tech Stack

- Python
- OpenCV
- MediaPipe
- Scikit-learn
- NumPy
- TextBlob
- Computer Vision
- Machine Learning
  
---

## 📁 Directory Structure

```
├── collect_imgs.py          # Step 1: Collect gesture images
├── create_dataset.py        # Step 2: Generate dataset with hand landmarks
├── train_classifier2.py      # Step 3: Train Random Forest classifier
├── inference5.py            # Step 4: Real-time gesture recognition and text output
├── data/                    # Folder to store gesture image folders (A-Z, DEL)
├── data.pickle              # Serialized feature-label dataset
├── model2.p                  # Trained classifier model
└── README.md
```

---

## 🔧 Setup Instructions

### ✅ Requirements

- Python 3.8+
- OpenCV
- MediaPipe
- scikit-learn
- numpy
- TextBlob

### 📦 Install Dependencies

```bash
pip install opencv-python mediapipe scikit-learn textblob numpy
python -m textblob.download_corpora
```

---

## 🚀 How to Run

### 1. Collect Gesture Images

Update `number_of_classes` and character index (`range(26, 27)` for 'Z' or change range for multiple letters) in `collect_imgs.py`:

```bash
python collect_imgs.py
```

Captured images are saved to the `data/` directory, one folder per letter.

---

### 2. Create Dataset

Extract landmarks using MediaPipe and serialize the dataset:

```bash
python create_dataset.py
```

Generates `data.pickle` containing features and labels.

---

### 3. Train the Classifier

Trains a `RandomForestClassifier` on the dataset:

```bash
python train_classifier2.py
```

Creates `model2.p` with the trained model.

---

### 4. Run Inference

Starts real-time hand detection, predicts letters, and forms corrected words/sentences:

```bash
python inference5.py
```

- Make gestures in front of the webcam.
- Predicted letters will appear and form a word.
- Once the hand disappears for ~2 seconds, the word is auto-corrected using TextBlob and added to the sentence.

---

## ✨ Features

- Real-time ASL alphabet recognition.
- Error correction using **TextBlob** NLP.
- Automatic handling of **DELETE** gesture (`class 26`).
- Webcam interface with prediction bounding boxes.

---

## 📝 Notes

- You can adjust the `capture_delay` and `min_detection_confidence` in `inference5.py` to optimize performance.
- Ensure consistent hand position and lighting during image collection for better accuracy.
- Train with more images per class for robust performance.

---

## 🙌 Future Improvements

- Add support for full words or sentence-level gesture recognition.
- Integrate speech synthesis for audible feedback.
- Use deep learning models (e.g., CNN + LSTM) for better accuracy and temporal understanding.

---
## Author

