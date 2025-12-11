# 라우팅 모듈
from .main_routes import bp as main_bp
from .vision_routes import bp as vision_bp
from .rnn_routes import bp as rnn_bp


# 블루프린트 리스트로 관리
blueprints = [
    main_bp,
    vision_bp,
    rnn_bp
]