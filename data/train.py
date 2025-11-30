import subprocess
import os

# train.py 기준 절대 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 학습 스크립트 목록 (여러 개 가능)
TRAIN_SCRIPTS = [
    # os.path.join(BASE_DIR, "models", "img_classifier.py"),
    os.path.join(BASE_DIR, "models", "rnn_simple_conversation.py"),
    # os.path.join(BASE_DIR, "models", "other_model.py"),  # 추가 가능
]

if __name__ == "__main__":
    for script in TRAIN_SCRIPTS:
        if not os.path.exists(script):
            print(f"[WARN] 학습 스크립트를 찾을 수 없습니다: {script}")
            continue

        print(f"[INFO] '{script}' 실행 중...")
        try:
            # 로그 파일 저장
            log_file = os.path.join(BASE_DIR, "..", "logs", os.path.basename(script) + ".log")
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

            subprocess.run(
                ["python", script],
                check=True,
                stdout=open(log_file, "w"),
                stderr=subprocess.STDOUT
            )

            print(f"[INFO] '{script}' 학습 완료! 로그: {log_file}\n")

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] '{script}' 학습 실패: {e}\n")