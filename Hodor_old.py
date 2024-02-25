import cv2
import face_recognition
import os
import numpy as np
from pathlib import Path
import pickle


DEFAULT_ENCODINGS_PATH = Path("Encodings/encodings.pkl")

# # Function to load and encode faces from the uploaded_images directory
# def load_and_encode_faces(base_dir='training_data'):
#     known_face_encodings = []
#     known_face_names = []
#     # Loop through each person's folder
#     for person_name in os.listdir(base_dir):
#         person_dir = os.path.join(base_dir, person_name)
#         if os.path.isdir(person_dir):
#             for filename in os.listdir(person_dir):
#                 if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
#                     filepath = os.path.join(person_dir, filename)
#                     print(f"Processing image: {filepath}")
#                     image = face_recognition.load_image_file(filepath)
#                     try:
#                         encoding = face_recognition.face_encodings(image)[0]
#                         known_face_encodings.append(encoding)
#                         known_face_names.append(person_name)
#                         print(f"Face encoded for: {person_name}")
#                     except IndexError:
#                         print(f"No face found in image: {filepath}")
#     return known_face_encodings, known_face_names

# # Load encoded faces
# known_face_encodings, known_face_names = load_and_encode_faces()

encodings_location=DEFAULT_ENCODINGS_PATH

with encodings_location.open(mode="rb") as f:
    loaded_encodings = pickle.load(f)

known_face_encodings, known_face_names = loaded_encodings["encodings"], loaded_encodings["names"]               

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])  # Convert BGR (OpenCV) to RGB (face_recognition)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the detected face matches any known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        #Print the name of the person in the frame
        print(name)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()