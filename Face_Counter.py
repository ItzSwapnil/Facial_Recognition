'''
Using DeepFace to count the number of faces in the frame
Smoother than 3rd Sem
'''

import cv2
from deepface import DeepFace

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Perform face detection with enforce_detection set to False
    try:
        faces = DeepFace.extract_faces(frame, detector_backend='opencv', enforce_detection=False)
        # Filter valid faces (e.g., using a confidence threshold if available)
        num_faces = sum(1 for face in faces if face['confidence'] >= 0.9)  # Adjust threshold as needed
    except:
        num_faces = 0

    # Display the number of faces
    cv2.putText(frame, f'Number of Persons: {num_faces}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Webcam', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
