#!/usr/bin/env python3
"""
Data Matrix Decoder Web Server
Flask-based web service for data matrix decoding
"""

import os
import json
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from src.data_matrix_decoder import decode_datamatrix_from_image, process_image
from src.utils import draw_results
import cv2

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Create directories
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path(app.config['RESULTS_FOLDER']).mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
def index():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'service': 'Data Matrix Decoder',
        'version': '1.0.0'
    })


@app.route('/decode', methods=['POST'])
def decode_image():
    """Decode data matrix from uploaded image"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Get parameters
        timeout_ms = request.form.get('timeout', 2000, type=int)
        draw_boxes = request.form.get('draw_boxes', 'false').lower() == 'true'

        # Process image
        results = decode_datamatrix_from_image(filepath, timeout_ms=timeout_ms)
        
        response_data = {
            'filename': filename,
            'codes_found': len(results),
            'results': []
        }

        for i, (data, box) in enumerate(results):
            x, y, w, h = box
            response_data['results'].append({
                'index': i + 1,
                'data': data,
                'bounding_box': {
                    'x': x, 'y': y, 'width': w, 'height': h
                }
            })

        # Optionally create annotated image
        if draw_boxes and results:
            image = cv2.imread(filepath)
            annotated_path = os.path.join(
                app.config['RESULTS_FOLDER'], 
                f"annotated_{filename}"
            )
            draw_results(image, results, annotated_path)
            response_data['annotated_image'] = f"/download/{Path(annotated_path).name}"

        # Clean up uploaded file
        os.remove(filepath)

        return jsonify(response_data)

    except Exception as e:
        # Clean up on error
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500


@app.route('/download/<filename>')
def download_file(filename):
    """Download annotated image"""
    try:
        return send_file(
            os.path.join(app.config['RESULTS_FOLDER'], filename),
            as_attachment=True
        )
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404


@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large'}), 413


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    
    print(f"Starting Data Matrix Decoder Server on port {port}")
    print(f"Debug mode: {debug}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)