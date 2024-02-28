# import argparse
import cv2
import pickle
from pathlib import Path
import numpy as np
import face_recognition
from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO, emit
import time
import requests

import base64
import os
from email.mime.text import MIMEText
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

DEFAULT_ENCODINGS_PATH = Path("Encodings/encodings.pkl")

app = Flask(__name__)
socketio = SocketIO(app)
cap = cv2.VideoCapture(0)

def send_email_notification(message):
    SCOPES = [
            "https://www.googleapis.com/auth/gmail.send"
        ]

    creds = None

    if os.path.exists("credentials.json"):
        try:
            creds = Credentials.from_authorized_user_file("credentials.json", SCOPES)
            print(creds)
        except Exception as e:
            print(e)
            creds = None
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open("credentials.json", "w") as token:
            token.write(creds.to_json())

    service = build('gmail', 'v1', credentials=creds)
    message = MIMEText(message)
    message['to'] = 'td32@cttd.biz'
    message['subject'] = 'New Face(s) Detected by Hodor'
    create_message = {'raw': base64.urlsafe_b64encode(message.as_bytes()).decode()}

    try:
        message = (service.users().messages().send(userId="me", body=create_message).execute())
        print(F'sent message to {message} Message Id: {message["id"]}')
    except HttpError as error:
        print(F'An error occurred: {error}')
        message = None

# parser = argparse.ArgumentParser(description="Recognize faces in a live video stream")
# parser.add_argument("--train", action="store_true", help="Train on input data")
# parser.add_argument("--test", action="store_true", help="Test the model with live video stream")
# parser.add_argument("-m", action="store", default="hog", choices=["hog", "cnn"], help="Which model to use for training: hog (CPU), cnn (GPU)")
# args = parser.parse_args()

class RateLimiter:
    def __init__(self, interval):
        """
        Initialize the rate limiter.
        :param interval: Minimum time interval between actions in seconds.
        """
        self.interval = interval
        self.last_action_time = 0

    def allow_action(self):
        """
        Check if an action is allowed based on the interval since the last action.
        :return: True if the action is allowed, False otherwise.
        """
        current_time = time.time()
        if current_time - self.last_action_time >= self.interval:
            self.last_action_time = current_time
            return True
        return False

def recognize_live_faces(model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH) -> None:
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    while (True):
        names_detected = []

        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break
        
        rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])

        input_face_locations = face_recognition.face_locations(rgb_frame, model=model)
        input_face_encodings = face_recognition.face_encodings(rgb_frame, input_face_locations)

        for bounding_box, unknown_encoding in zip(input_face_locations, input_face_encodings):
            name, confidence = _recognize_face(unknown_encoding, loaded_encodings)
            if confidence is not None:
                confidence_text = f"{name} ({(1-confidence)*100:.2f}%)"
            else:
                confidence_text = name

            top, right, bottom, left = bounding_box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, confidence_text, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
            names_detected.append(name)
            
        # if names_detected and notification_limiter.allow_action():
        #     send_pushover_notification(f"People detected at the front door: {names_detected}")

        # cv2.imshow('Video Feed', frame)
        # print(name)
        socketio.emit('detected_names', names_detected)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
        # key = cv2.waitKey(20)
        # if key == 27:
        #     break
    
    cap.release()
    cv2.destroyAllWindows()

def _recognize_face(unknown_encoding, loaded_encodings):
    distances = face_recognition.face_distance(loaded_encodings["encodings"], unknown_encoding)
    if len(distances) == 0:
        return "Unknown", None

    best_match_index = np.argmin(distances)
    if distances[best_match_index] < 0.5:  # Adjust threshold as needed
        name = loaded_encodings["names"][best_match_index]
        confidence = distances[best_match_index]
        return name, confidence
    else:
        return "Unknown", None
    
# Example usage with a 5-minute interval (300 seconds)
notification_limiter = RateLimiter(60)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(recognize_live_faces(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detected_names')
def detected_names():
    global names_detected
    return jsonify(names_detected)

if __name__ == "__main__":
    # recognize_live_faces()
    socketio.run(app, host='0.0.0.0', port=3000, debug=True, use_reloader=False)
    # if args.train:
    #     print("Training mode selected. This option is not available in the live stream.")
    # elif args.test:
    #     # app.run(host='0.0.0.0', port=3000, debug=True)
    #     recognize_live_faces()
    # else:
        # app.run(host='0.0.0.0', port=3000, debug=True, use_reloader=False)
    #     # recognize_live_faces()

