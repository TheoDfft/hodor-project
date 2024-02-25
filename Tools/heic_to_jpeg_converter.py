import os
import pyheif
from PIL import Image

# Function to convert HEIC image to JPEG
def convert_heic_to_jpg(heic_path, jpg_path):
    # Read the HEIC file
    heif_file = pyheif.read(heic_path)
    
    # Create an image object from the HEIC data
    image = Image.frombytes(
        heif_file.mode, 
        heif_file.size, 
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride,
    )
    
    # Save the image as JPEG
    image.save(jpg_path, format="JPEG")

    # Delete the original HEIC file
    os.remove(heic_path)
    

# Directory containing the training data
training_data_dir = 'training_data'

# Traverse through the directory and its subdirectories
for root, dirs, files in os.walk(training_data_dir):
    for file in files:
        # Check if the file has .HEIC or .heic extension
        if file.endswith('.HEIC') or file.endswith('.heic'):
            # Get the full path of the HEIC file
            heic_path = os.path.join(root, file)
            
            # Generate the corresponding JPEG file path
            jpg_path = os.path.splitext(heic_path)[0] + '.jpg'
            
            # Convert the HEIC file to JPEG
            convert_heic_to_jpg(heic_path, jpg_path)