FROM python:3.9-slim

WORKDIR /usr/src/app

# 필수 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 실행
# 컨테이너 시작 시 train.py 실행 후 Flask 실행
CMD ["bash", "-c", "python /usr/src/data/train.py && python /usr/src/app/app.py"]
# CMD ["python", "app.py"]