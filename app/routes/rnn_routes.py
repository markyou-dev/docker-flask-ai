from flask import Blueprint, render_template, request, jsonify, current_app

bp = Blueprint("rnn", __name__, url_prefix="/rnn")

@bp.route('/')
def index():
    return render_template('rnn/index.html')

@bp.route("/predict", methods=["POST"])
def predict_rnn():
    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"error": "message field is required"}), 400

    user_text = data["message"]

    # 모델 호출
    try:
        reply = current_app.rnn.predict(user_text)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "input": user_text,
        "response": reply
    })
