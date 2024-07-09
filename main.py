'''
Face Detection -> Facial Recognition
3rd Semester -> 4th Semester

slower yet quite accurate
'''


import cv2
import face_recognition
import os

face_dataset_dir = "face_dataset"

# name loader
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

# open webcam
video_capture = cv2.VideoCapture(0)


#0.5-0.6 for my webcam
tolerance = 0.5

while True:
    # each frame
    ret, frame = video_capture.read()

    # Convert BGR to RGB color (opencv to face_recognition)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # search faces
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # scan each frame for faces
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=tolerance)
        name = "Unknown"

        # If a match was found in known_face_encodings, use the first one
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = face_distances.argmin()
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 180, 80), 2)

        # label the faces
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (255, 144, 90), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display frame by frame vid
    cv2.imshow('Video', frame)

    # quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
