from flask import Blueprint, request, render_template, current_app
import os
from datetime import datetime

bp = Blueprint('vision', __name__, url_prefix='/vision')

@bp.route('/')
def index():
    return render_template('vision/index.html')

@bp.route('/result', methods=['POST'])
def result():
    if 'file' not in request.files or request.files['file'].filename == '':
        return "No file uploaded", 400

    file = request.files['file']
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    result, confidence = current_app.classifier.predict(filepath)
    return render_template(
        'vision/result.html', 
        result=result, 
        confidence=confidence, 
        image_path=f"/image/{filename}"
    )
