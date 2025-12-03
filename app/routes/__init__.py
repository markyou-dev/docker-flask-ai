# 라우팅 모듈
from .main_routes import bp as main_bp
from .predict_routes import bp as predict_bp

# 블루프린트 리스트로 관리
blueprints = [
    main_bp,
    predict_bp
]