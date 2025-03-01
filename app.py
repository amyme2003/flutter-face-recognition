from flask import Flask, request, jsonify
import os
import cv2
import numpy as np

app = Flask(__name__)

dataset_path = "dataset/"
trainer_path = "trainer.yml"

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Ensure dataset directory exists
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

@app.route('/upload_faces', methods=['POST'])
def upload_faces():
    """Receive multiple images for a person and store them."""
    person_name = request.form.get('name')
    if not person_name:
        return jsonify({"error": "Name is required"}), 400
    
    person_dir = os.path.join(dataset_path, person_name)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)
    
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images received"}), 400
    
    for file in files:
        img_path = os.path.join(person_dir, file.filename)
        file.save(img_path)

    return jsonify({"message": "Images saved for " + person_name}), 200

@app.route('/train', methods=['GET'])
def train_model():
    """Train the LBPH recognizer with stored images."""
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, labels = [], []
    label_map = {}
    label_count = 0

    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)

        # Skip non-directory files like .gitkeep
        if not os.path.isdir(person_dir):
            continue

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)

            # Ensure only image files are processed
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (200, 200))  # Ensure consistent image size
            faces.append(img)

            if person_name not in label_map:
                label_map[person_name] = label_count
                label_count += 1

            labels.append(label_map[person_name])

    if not faces:
        return jsonify({"error": "No valid images found for training"}), 400

    recognizer.train(faces, np.array(labels))
    recognizer.save(trainer_path)

    return jsonify({"message": "Training complete", "people_trained": list(label_map.keys())}), 200

