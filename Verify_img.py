'''
Check every img has a face in it
(cant detect sideways faces)
'''


import os
import face_recognition

face_dataset_dir = "face_dataset"

known_face_encodings = []
known_face_names = []

for name in os.listdir(face_dataset_dir):
    for filename in os.listdir(f"{face_dataset_dir}/{name}"):
        image_path = f"{face_dataset_dir}/{name}/{filename}"
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(name)
        else:
            print(f"No face found in image {image_path}")

print("Everything is ok")
