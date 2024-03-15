# This code sets up a Flask web application and defines a function for checking if a file has an allowed extension.

from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import wand.image as WandImage

UPLOAD_FOLDER = 'training_data'  # The folder where uploaded images will be stored
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg','heic'}  # The set of allowed file extensions

app = Flask(__name__)  # Create a Flask application instance
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  # Configure the upload folder for the application

def allowed_file(filename):
    # Check if the filename has an extension and if it is allowed
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#method to convert heic to jpg
def heic_to_jpg(heic_path, jpg_path):
    with WandImage(filename=heic_path) as img:
        img.format = 'jpg'
        img.save(filename=jpg_path)

@app.route('/', methods=['GET', 'POST'])
def upload_files():
    """
    This function handles the file upload process in a Flask web application.
    It checks if the HTTP request method is POST and if the request contains files and a person_name.
    If the conditions are met, it saves the uploaded files to a specified folder with the person_name as a subfolder.
    Finally, it redirects the user to the same upload page after the files are saved.
    If the request method is GET, it renders the upload.html template.
    """
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'files[]' not in request.files:
            return redirect(request.url)
        files = request.files.getlist('files[]')
        person_name = request.form['person_name']
        if not files or person_name == '':
            return redirect(request.url)
        
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], person_name)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                #check if the file is heic and convert it to jpg if it is
                if filename.lower().endswith("heic"):
                    heic_path = os.path.join(save_path, filename)
                    jpg_path = os.path.join(save_path, filename.replace("HEIC", "jpg"))
                    file.save(heic_path)
                    heic_to_jpg(heic_path, jpg_path)
                    os.remove(heic_path)
                else:
                    file.save(os.path.join(save_path, filename))
        return redirect(url_for('upload_files'))
    return render_template('upload_page.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
