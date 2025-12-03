FROM python:3.9-slim

WORKDIR /usr/src

# 필수 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 파일경로 복사
COPY . .

# 실행
# 컨테이너 시작 시 train.py 실행 후 Flask 실행
CMD ["bash", "-c", "python ./data/train.py && flask run --host=0.0.0.0 --port=$FLASK_PORT"]
