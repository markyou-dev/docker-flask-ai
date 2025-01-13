from flask import Flask, request, jsonify, render_template, send_file, abort
import os
from datetime import datetime
from models import ImgClassifier

class Config:
    PORT = int(os.environ.get('FLASK_PORT', 5000))
    DEBUG = bool(int(os.environ.get('FLASK_DEBUG', 1)))
    UPLOAD_FOLDER = '/usr/src/uploads'
    MODEL_PATH = '/usr/src/data/save_models/img_classifier.h5'

app = Flask(__name__)
app.config.from_object(Config)

# 업로드 디렉토리 생성
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# AI 모델 초기화
classifier = ImgClassifier(app.config['MODEL_PATH'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image/<filename>')
def serve_image(filename):
    # 파일의 절대경로 생성
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # 파일 존재 여부 확인
    if not os.path.isfile(file_path):
        abort(404, description="File not found")
    
    # 파일 반환
    return send_file(file_path, mimetype='image/jpeg')  # MIME 타입은 이미지 형식에 맞게 설정

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # 파일 저장
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    result, confidence = classifier.predict(filepath)

    return render_template(
        'predict.html',
        result=result,
        confidence=confidence,
        image_path=f"/image/{filename}"
    )
    
    # try:
    #     # 예측 수행
    #     result, confidence = classifier.predict(filepath)
    #     return jsonify({
    #         'result': result,
    #         'confidence': float(confidence)
    #     })
    # except Exception as e:
    #     return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", 
            port=app.config['PORT'],
            debug=app.config['DEBUG'])
