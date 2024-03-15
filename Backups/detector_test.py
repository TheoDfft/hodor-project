import argparse
import pickle
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
import face_recognition
from PIL import Image, ImageDraw
import time
from flask import Flask, render_template, Response

DEFAULT_ENCODINGS_PATH = Path("Encodings/encodings.pkl")
BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"

app = Flask(__name__)
cap = cv2.VideoCapture('file_example_MP4_640_3MG.mp4')

# # Create directories if they don't already exist
# Path("training").mkdir(exist_ok=True)
# Path("output").mkdir(exist_ok=True)
# Path("validation").mkdir(exist_ok=True)

# parser = argparse.ArgumentParser(description="Recognize faces in an image")
# parser.add_argument("--train", action="store_true", help="Train on input data")
# parser.add_argument(
#     "--validate", action="store_true", help="Validate trained model"
# )
# parser.add_argument(
#     "--test", action="store_true", help="Test the model with an unknown image"
# )
# parser.add_argument(
#     "-m",
#     action="store",
#     default="hog",
#     choices=["hog", "cnn"],
#     help="Which model to use for training: hog (CPU), cnn (GPU)",
# )
# parser.add_argument(
#     "-f", action="store", help="Path to an image with an unknown face"
# )
# args = parser.parse_args()


def encode_known_faces(model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH) -> None:
    """
    Loads images in the training directory and builds a dictionary of their
    names and encodings.
    """

    names = []
    encodings = []

    for filepath in Path("training_data").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)


def recognize_live_faces(model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH,) -> None:
    """
    Given an unknown image, get the locations and encodings of any faces and
    compares them against the known encodings to find potential matches.
    """

    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    # cap = cv2.VideoCapture(0)

    process_every_nth_frame = 1  # Process every 5th frame to reduce workload
    frame_count = 0
    
    #We are going to loop through the live video feed and process every 5th frame

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame_count += 1
        if frame_count % process_every_nth_frame != 0:
            cv2.imshow("Video Feed", frame)
            continue
        
        #First we setup our time counter to calculate how long it takes to process each frame
        start_time = time.time()

        # Optionally resize frame for faster face recognition processing
        # rgb_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])  # Convert BGR (OpenCV) to RGB (face_recognition)
        
        input_face_locations = face_recognition.face_locations(rgb_frame, model=model)

        input_face_encodings = face_recognition.face_encodings(rgb_frame, input_face_locations)

        for bounding_box, unknown_encoding in zip(input_face_locations, input_face_encodings):

            name, confidence = _recognize_face(unknown_encoding, loaded_encodings)

            if confidence is not None:
                if confidence <= 1 and confidence >= 0:
                    confidence = (1-confidence)*100
                else:
                    confidence = 0
                confidence_text = f"{name} ({confidence:.2f}%)"
            else:
                confidence_text = name


            top, right, bottom, left = bounding_box
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            # top *= 2
            # right *= 2
            # bottom *= 2
            # left *= 2

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, confidence_text, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # cv2.imshow("Video Feed", frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Calculate the time it took to process the frame
        print("Process Time: ", time.time() - start_time)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def _recognize_face(unknown_encoding, loaded_encodings):
    """
    Given an unknown encoding and all known encodings, finds the known
    encoding with the most matches and returns both the name and the
    confidence (distance) of the recognition.
    """
    # Compute the face distance between the unknown face and all known faces
    distances = face_recognition.face_distance(
        loaded_encodings["encodings"], unknown_encoding
    )
    
    if len(distances) == 0:
        return "Unknown", None  # No matches found
    
    # Find the known face with the smallest distance to the unknown face
    best_match_index = np.argmin(distances)
    if distances[best_match_index] < 0.5:  # Adjust threshold as needed
        name = loaded_encodings["names"][best_match_index]
        confidence = distances[best_match_index]
        return name, confidence
    else:
        return "Unknown", None


    # """
    # Given an unknown encoding and all known encodings, find the known
    # encoding with the most matches.
    # """
    # boolean_matches = face_recognition.compare_faces(
    #     loaded_encodings["encodings"], unknown_encoding
    # )
    # votes = Counter(
    #     name
    #     for match, name in zip(boolean_matches, loaded_encodings["names"])
    #     if match
    # )
    # if votes:
    #     return votes.most_common(1)[0][0]


# def _display_face(draw, bounding_box, name):
#     """
#     Draws bounding boxes around faces, a caption area, and text captions.
#     """
#     top, right, bottom, left = bounding_box
#     draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR)
#     text_left, text_top, text_right, text_bottom = draw.textbbox(
#         (left, bottom), name
#     )
#     draw.rectangle(
#         ((text_left, text_top), (text_right, text_bottom)),
#         fill=BOUNDING_BOX_COLOR,
#         outline=BOUNDING_BOX_COLOR,
#     )
#     draw.text(
#         (text_left, text_top),
#         name,
#         fill=TEXT_COLOR,
#     )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(recognize_live_faces(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

app.run(host='0.0.0.0', port=3000, debug=True)
# recognize_live_faces()
# if __name__ == "__main__":
#     if args.train:
#         encode_known_faces(model=args.m)
#     if args.test:
#         recognize_live_faces(image_location=args.f, model=args.m)
