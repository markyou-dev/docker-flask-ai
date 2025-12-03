from flask import Blueprint, request, render_template, current_app, send_file, abort
import os
from datetime import datetime

bp = Blueprint('predict', __name__)

@bp.route('/image/<filename>')
def serve_image(filename):
    file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    if not os.path.isfile(file_path):
        abort(404)
    return send_file(file_path, mimetype='image/jpeg')

@bp.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or request.files['file'].filename == '':
        return "No file uploaded", 400

    file = request.files['file']
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    result, confidence = current_app.classifier.predict(filepath)
    return render_template('predict.html', result=result, confidence=confidence, image_path=f"/image/{filename}")
