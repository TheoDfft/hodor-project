import argparse
import cv2
import pickle
from pathlib import Path
import numpy as np
import face_recognition
from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO, emit
import time
import requests
# from sqlalchemy import create_engine, Column, String, DateTime
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker
# from sqlalchemy import Integer
from datetime import datetime, timedelta
from pushover import Client
from kasa_utils import ToggleKasaPlug
from db_utils import Detection, get_db_session

# Base = declarative_base()
# class Detection(Base):
#     __tablename__ = 'hodor_detections'
#     id = Column(Integer, primary_key=True, autoincrement=True)
#     name = Column(String, nullable=False)
#     detected_at = Column(DateTime, default=datetime.now())
# # Create an SQLite engine
# engine = create_engine('sqlite:///hodor_detections.db')
# # Create all tables
# Base.metadata.create_all(engine)
# # Create a sessionmaker
# Session = sessionmaker(bind=engine)

# def get_db_session():
#     session = Session()
#     return session

DEFAULT_ENCODINGS_PATH = Path("Encodings/encodings.pkl")
DETECTION_PERIOD_SECONDS = 300
NUMBER_OF_FRAMES_FOR_CONFIDENCE = 20
DETECTION_THRESHOLD = 0.25

last_detections = []
detections_data = []

app = Flask(__name__)
socketio = SocketIO(app)
light_toggler = ToggleKasaPlug()
po_client = Client("u3mra51amezcopes8nx5csnkm2xn11", api_token="acqnqxckdz3hhaqkn1gebpj2vq9hcb")
# cap = cv2.VideoCapture(0)

with get_db_session() as session:
    since = datetime.now() - timedelta(hours=24)
    detections = session.query(Detection).filter(Detection.detected_at > since).all()
    detections_data = [{"Name": detection.name, "Detected_at": detection.detected_at.strftime("%Y-%m-%d %H:%M PST")} for detection in detections]
    
    socketio.emit('detections_last_24_hours', detections_data)

parser = argparse.ArgumentParser(description="Recognize faces in a live video stream")
parser.add_argument("--headless", action="store_true", help="Run the app without local server or camera stream")
parser.add_argument("--server", action="store_true", help="Run the app on a local server (http://192.168.1.34:3000/)")
args = parser.parse_args()

def door_monitor():
    frame_delay = 30
    change_threshold_percent = 1

    light_toggler.modifyKasaDeviceState(0)
    cap = cv2.VideoCapture(0)
    
    # Read the first frame to initialize the background
    _, background = cap.read()
    background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    frame_counter = 0
    while True:
        frame_counter += 1

        _, frame = cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_counter % frame_delay == 0:


            # Compute the absolute difference between the current frame and background
            frame_diff = cv2.absdiff(background, gray_frame)

            # Apply a threshold to get a binary image
            _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
            
            changed_pixels = cv2.countNonZero(thresh)
            total_pixels = thresh.size
            changed_percent = (changed_pixels / total_pixels) * 100

            if changed_percent > change_threshold_percent:
                print("Image changed by: ", changed_percent, "%")
                cap.release()
                #cv2.destroyAllWindows()
                light_toggler.modifyKasaDeviceState(1)
                return

            # # Display the thresholded image
            # cv2.imshow('Door Monitor - Frame Difference', thresh)

            # Update the background periodically or based on some condition
            # For continuous monitoring, you might want to update the background less frequently
            # or use more sophisticated methods to handle gradual lighting changes.
            background = gray_frame
        # Press 'q' to exit the loop
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #      break
    
def update_faces_detected(names_detected_live, frame):
    global last_detections
    global detections_data

    confirmed_detections = []

    for name in names_detected_live:
        if len(last_detections) < NUMBER_OF_FRAMES_FOR_CONFIDENCE:
            last_detections.append(name)
            return
        last_detections.pop(0)
        last_detections.append(name)

    #add the unique names that appear more than 1/4th of the time in the last 30 frames
    for name in last_detections:
        if (last_detections.count(name) >= NUMBER_OF_FRAMES_FOR_CONFIDENCE * DETECTION_THRESHOLD) and (name not in confirmed_detections) and (name != "None"):
            confirmed_detections.append(name)
    #sort the names in confirmed_detection by the first letter of the name
    confirmed_detections.sort()

    
    with get_db_session() as session:
        for name in confirmed_detections:
            # Check if the name was detected in the last 10 minutes
            recent_detection = session.query(Detection).filter(Detection.name == name, Detection.detected_at > datetime.now() - timedelta(minutes=10)).first()
            if not recent_detection:
                # Add new detection to the database
                new_detection = Detection(name=name, detected_at=datetime.now())
                session.add(new_detection)

                _ , buffer = cv2.imencode('.jpg', frame)
                po_image = buffer.tobytes()
                detection_time = datetime.now().strftime("%H:%M PST")

                po_client.send_message(f'{name} has been detected at {detection_time}', attachment=po_image, title="Hodor detected someone a the door!")

        session.commit()

        since = datetime.now() - timedelta(hours=24)
        detections = session.query(Detection).filter(Detection.detected_at > since).all()
        detections_data = [{"Name": detection.name, "Detected_at": detection.detected_at.strftime("%Y-%m-%d %H:%M PST")} for detection in detections]
        
        socketio.emit('detections_last_24_hours', detections_data)
    
    socketio.emit('detected_names', confirmed_detections)

def recognize_live_faces(model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH):
    start_time = time.time()
    print("Entered recognize_live_faces")
    cap = cv2.VideoCapture(0)
    
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    while (time.time() - start_time) < DETECTION_PERIOD_SECONDS:
        names_detected_live = []

        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break
        
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
        

        input_face_locations = face_recognition.face_locations(rgb_frame, model=model)
        input_face_encodings = face_recognition.face_encodings(rgb_frame, input_face_locations)

        for bounding_box, unknown_encoding in zip(input_face_locations, input_face_encodings):
            name, confidence = _recognize_face(unknown_encoding, loaded_encodings)
            if confidence is not None:
                confidence_text = f"{name} ({(1-confidence)*100:.2f}%)"
            else:
                confidence_text = name

            top, right, bottom, left = bounding_box

            #readjust the size of the bounding box by multiplying the coordinate by 4 
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, confidence_text, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
            names_detected_live.append(name)
            
        if not names_detected_live:
            update_faces_detected(["None"], None)
        else:
            update_faces_detected(names_detected_live, frame)

        # cv2.imshow('Video Feed', frame)

        if args.server:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # save_known_faces()
            break

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(recognize_live_faces(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detected_names')
def detected_names(confirmed_detections):
    return jsonify(confirmed_detections)

@app.route('/detections_last_24_hours')
def detections_last_24_hours():
    global detections_data 
    return jsonify(detections_data)

if __name__ == "__main__":
    try:
        while True:
            # door_monitor()
            if args.headless:
                recognize_live_faces()
            elif args.server:
                socketio.run(app, host='0.0.0.0', port=3000, debug=True, use_reloader=False)
                break  # Exiting the loop since socketio.run is blocking
            else:
                recognize_live_faces()
    except KeyboardInterrupt:
        print("Exiting due to Ctrl+C")

