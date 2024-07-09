'''
faster, smoother and securer than previous model
requires better webcam to work flawlessly
'''


import os
import cv2
import numpy as np
import insightface
import json

# Load face embeddings
def load_face_embeddings(dataset_path):
    face_encodings = {}
    for embeddings_file in os.listdir(dataset_path):
        if embeddings_file.endswith('_embeddings.json'):
            person_name = embeddings_file.split('_embeddings.json')[0]
            embeddings_path = os.path.join(dataset_path, embeddings_file)
            with open(embeddings_path, 'r') as f:
                embeddings_data = json.load(f)
                embeddings = [np.array(embed['embedding']) for embed in embeddings_data]
                face_encodings[person_name] = embeddings
    return face_encodings

# best match finder
def find_best_match(face_encoding, face_encodings, threshold=22.2):
    best_match = None
    best_distance = float('inf')
    for person_name, encodings in face_encodings.items():
        for stored_encoding in encodings:
            distance = np.linalg.norm(stored_encoding - face_encoding)
            print(f"Distance to {person_name}: {distance}")
            if distance < best_distance:
                best_distance = distance
                best_match = person_name
    print(f"Best match: {best_match} with distance: {best_distance}")
    return best_match if best_distance < threshold else "Unknown"


# InsightFace model
def initialize_insightface_model():
    model = insightface.app.FaceAnalysis()
    model.prepare(ctx_id=0, det_size=(640, 480))
    return model

#using embeddings for real time face recognition
def real_time_face_recognition(model, face_encodings):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = model.get(frame)
        if faces:
            for face in faces:
                # Normalize face embedding
                face_embedding = face.embedding / np.linalg.norm(face.embedding)
                # Compare both embeddings
                match = find_best_match(face_embedding, face_encodings)
                print(f"Recognized: {match}")  # Debug print
                # Draw bounding box and display name
                bbox = face.bbox.astype(int)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(frame, match, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow('Face Recognition', frame)
        print("Display updated")  # Debug print
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        # Load faces embeddings dataset
        dataset_path = "embeddings"
        face_encodings = load_face_embeddings(dataset_path)
        print(f"Loaded face encodings: {face_encodings.keys()}")

        model = initialize_insightface_model()

        # real-time face recognition
        real_time_face_recognition(model, face_encodings)

    except Exception as e:
        print(f"Error: {e}")
