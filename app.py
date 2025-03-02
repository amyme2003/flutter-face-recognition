from flask import Flask, request, jsonify
import os
import cv2
import numpy as np

app = Flask(__name__)

dataset_path = "/app/dataset/"  # Store datasets in the app folder
trainer_path = "/app/trainer.yml"  # Store trained model inside the app directory

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Ensure dataset directory exists
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Face Recognition API is running"}), 200

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

    return jsonify({"message": f"Images saved for {person_name}"}), 200

@app.route('/train', methods=['GET'])
def train_model():
    """Train the LBPH recognizer with stored images."""
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, labels = [], []
    label_map = {}
    label_count = 0

    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_dir):
            continue

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (200, 200))
            faces.append(img)

            if person_name not in label_map:
                label_map[person_name] = label_count
                label_count += 1

            labels.append(label_map[person_name])

    if not faces:
        return jsonify({"error": "No valid images found for training"}), 400

    recognizer.train(faces, np.array(labels))
    recognizer.save(trainer_path)  # Save model inside /app

    return jsonify({"message": "Training complete", "people_trained": list(label_map.keys())}), 200

@app.route('/detect', methods=['POST'])
def detect_face():
    """Detect a face in an uploaded image and return the closest trained match."""
    if not os.path.exists(trainer_path):
        return jsonify({"error": "Model not trained yet"}), 400

    file = request.files['image']
    nparr = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(trainer_path)  # Load trained model

    faces = face_detector.detectMultiScale(gray, 1.2, 5)
    if len(faces) == 0:
        return jsonify({"message": "No face detected"}), 400

    label_map = {idx: name for idx, name in enumerate(os.listdir(dataset_path))}
    best_match = "Unknown"
    best_confidence = float("inf")

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))
        label, confidence = recognizer.predict(face)

        if confidence < best_confidence:
            best_confidence = confidence
            best_match = label_map.get(label, "Unknown")

    return jsonify({"recognized": best_match, "confidence": best_confidence}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=True)
