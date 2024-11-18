import cv2
from deepface import DeepFace
from collections import defaultdict
import time

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

models = ['Age', 'Gender', 'Emotion', 'Race']

# Preload models
for model in models:
    print(f"Preparing facial attribute model for {model}...")
    DeepFace.build_model(model, 'facial_attribute')

def get_face_identity(face_roi):
    known_faces = []
    try:
        # Compute embedding for the detected face
        embedding = DeepFace.represent(face_roi, model_name="Facenet")[0]["embedding"]
        for idx, known_face in enumerate(known_faces):
            # Compare embeddings
            if DeepFace.verify(face_roi, known_face)["verified"]:
                return idx  # Return existing face ID
        # If no match is found, add new face and assign new ID
        known_faces.append(face_roi)
        return len(known_faces) - 1
    except Exception as e:
        print(f"Error during face recognition: {e}")
        return None

def get_facial_attributes(file_path, fps=5):
    print(f"Analyzing facial attributes at {fps} fps")
    # Initialize video capture (default webcam)
    cap = cv2.VideoCapture(file_path)
    prev = 0
    attribute_counts = {}
    while cap.isOpened():
        # Read a frame from the video
        time_elapsed = time.time() - prev
        ret, frame = cap.read()
        if not ret:
            break

        if time_elapsed <= 1./fps:
            continue
        prev = time.time()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # For each detected face
        for (x, y, w, h) in faces:
            # Extract the face ROI (Region of Interest)
            face_roi = frame[y:y+h, x:x+w]
            face_id = get_face_identity(face_roi)
            if face_id is None:
                continue

            # Analyze the face for emotion using DeepFace
            try:
                result = DeepFace.analyze(face_roi, actions=['emotion', 'age', 'gender', 'race'], silent=True)

                # Extract results
                facial_attributes = {
                    'emotion': result[0]['dominant_emotion'],
                    'age': result[0]['age'],
                    'gender': max(result[0]['gender'], key=result[0]['gender'].get),
                    'race': result[0]['dominant_race']
                }
                if face_id not in attribute_counts:
                    attribute_counts[face_id] = {
                        'emotion': defaultdict(int),
                        'age': defaultdict(int),
                        'gender': defaultdict(int),
                        'race': defaultdict(int)
                    }
                attribute_counts[face_id]['emotion'][facial_attributes['emotion']] += 1
                attribute_counts[face_id]['age'][facial_attributes['age']] += 1
                attribute_counts[face_id]['gender'][facial_attributes['gender']] += 1
                attribute_counts[face_id]['race'][facial_attributes['race']] += 1

            except:
                pass

        # Display the resulting frame
        # cv2.imshow('Facial Expression Recognition', frame)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

    for id, data in attribute_counts.items():
        for attr, counts in data.items():
            attribute_counts[id][attr] = dict(counts)
            if attr in {'age', 'gender'}:
                attribute_counts[id][attr] = max(data[attr], key=data[attr].get)
                continue
            total = sum(counts.values())
            for x, count in counts.items():
                attribute_counts[id][attr][x] = float(count / total)
            
    print(f"Done processing facial attributes for {file_path}")
    return attribute_counts

# get_facial_attributes('data/scenes/hunger_games_scene_2-Scene-008.mp4')