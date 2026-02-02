"""
Face Recognition using OpenCV
Author: Shaivi Verma
Description:
This project implements real-time face recognition using OpenCV and the LBPH (Local Binary Patterns Histograms) algorithm.
It detects faces using Haar Cascade and recognizes known faces trained from images.
"""

import cv2
import numpy as np
import os

# -----------------------------
# CONFIGURATION
# -----------------------------
DATASET_DIR = "dataset"   # Folder with subfolders per person
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
MODEL_PATH = "face_trainer.yml"

# -----------------------------
# LOAD FACE DETECTOR
# -----------------------------
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
recognizer = cv2.face.LBPHFaceRecognizer_create()

# -----------------------------
# TRAINING FUNCTION
# -----------------------------
def train_model():
    faces = []
    labels = []
    label_map = {}
    current_label = 0

    for person_name in os.listdir(DATASET_DIR):
        person_path = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_path):
            continue

        label_map[current_label] = person_name

        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if gray_image is None:
                continue

            faces_detected = face_cascade.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=5)
            for (x, y, w, h) in faces_detected:
                faces.append(gray_image[y:y+h, x:x+w])
                labels.append(current_label)

        current_label += 1

    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_PATH)
    return label_map

# -----------------------------
# REAL-TIME RECOGNITION
# -----------------------------
def recognize_faces(label_map):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(face_roi)

            name = label_map[label] if confidence < 70 else "Unknown"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    label_mapping = train_model()
    recognize_faces(label_mapping)
