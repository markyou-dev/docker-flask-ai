from flask import Blueprint, render_template, current_app, send_file, abort
import os

bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    menus = [
        {
            "title": "고양이/강아지 이미지 분류",
            "desc": "개 vs 고양이 이미지 분류 등 비전 모델을 테스트합니다.",
            "icon": "fas fa-camera",
            "url": "/vision"
        },
        {
            "title": "RNN 채팅",
            "desc": "RNN 대화형 학습모델로 채팅을 진행합니다.",
            "icon": "fas fa-robot",
            "url": "/rnn"
        },
    ]
    return render_template('index.html', menus=menus)

@bp.route('/image/<filename>')
def serve_image(filename):
    file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    if not os.path.isfile(file_path):
        abort(404)
    return send_file(file_path, mimetype='image/jpeg')