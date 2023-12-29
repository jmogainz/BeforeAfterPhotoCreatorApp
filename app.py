from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
from constants import *
from image_merging import merge_images
from clustering import cluster_images

app = Flask(__name__)

input_dir_var = "UPLOADS_FOLDER"
output_dir_var = "PROCESSING_FOLDER"

app.config[input_dir_var] = UPLOAD_FOLDER
app.config[output_dir_var] = PROCESSED_FOLDER

# Helper function to check file extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to render the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image uploads
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'images' not in request.files:
        return jsonify({'error': 'No file part'})

    files = request.files.getlist('images')

    for file in files:
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config[input_dir_var], filename))

    clusters = cluster_images(app.config[input_dir_var])

    # For demonstration, let's assume processed files are saved in PROCESSED_FOLDER
    # and return their download links
    processed_files = os.listdir(app.config[output_dir_var])
    processed_files_urls = [f'/download/{filename}' for filename in processed_files]

    return jsonify({'links': processed_files_urls})

# Route to download processed images
@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config[output_dir_var],
                               filename, as_attachment=True)

# Running the Flask app
if __name__ == '__main__':
    app.run(debug=True)
