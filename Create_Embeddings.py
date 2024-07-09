import os
import cv2
import numpy as np
import insightface
import json

def create_embeddings(person_name, images_folder, model):
    embeddings = []
    for img_name in os.listdir(images_folder):
        img_path = os.path.join(images_folder, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            faces = model.get(img)
            if faces:
                for face in faces:
                    embeddings.append({
                        'person_name': person_name,
                        'embedding': face.embedding.tolist()
                    })
    return embeddings

# Function to initialize InsightFace model
def initialize_insightface_model():
    model = insightface.app.FaceAnalysis()
    model.prepare(ctx_id=0, det_size=(640, 480))
    return model

# Function to save embeddings to a JSON file
def save_embeddings(embeddings, output_folder, person_name):
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"{person_name}_embeddings.json")
    with open(output_file, 'w') as f:
        json.dump(embeddings, f)

if __name__ == "__main__":
    try:
        # Path to the folder containing face image subfolders
        dataset_folder = "face_dataset"

        # Initialize InsightFace model
        model = initialize_insightface_model()

        # Iterate over each person's folder in the dataset
        for person_name in os.listdir(dataset_folder):
            person_folder = os.path.join(dataset_folder, person_name)
            if os.path.isdir(person_folder):
                # Create embeddings for the person
                embeddings = create_embeddings(person_name, person_folder, model)

                # Save embeddings to an embeddings folder
                embeddings_folder = "embeddings"
                save_embeddings(embeddings, embeddings_folder, person_name)

                print(f"Embeddings saved to {os.path.join(embeddings_folder, person_name + '_embeddings.json')}")

    except Exception as e:
        print(f"Error: {e}")
