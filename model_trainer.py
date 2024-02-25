import pickle
import face_recognition
from pathlib import Path


def encode_known_faces(model: str = "hog", encodings_location: Path = Path('Encodings/encodings.pkl')) -> None:
    # Initialize empty lists to store names and encodings
    names = []
    encodings = []

    # Iterate through all files in the training_data directory
    for filepath in Path("training_data").glob("*/*"):
        # Extract the name from the parent directory of the file
        name = filepath.parent.name

        # Load the image using face_recognition library
        image = face_recognition.load_image_file(filepath)

        # Detect face locations in the image
        face_locations = face_recognition.face_locations(image, model=model)

        # Encode the faces found in the image
        face_encodings = face_recognition.face_encodings(image, face_locations)

        # Append the name and encoding to the respective lists
        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

        print("Names:", names)
        # print("Encodings:", encodings)

    # Create a dictionary to store the names and encodings
    name_encodings = {"names": names, "encodings": encodings}

    # Save the dictionary as a pickle file
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)

    # Print the names and encodings for debugging purposes


encode_known_faces()