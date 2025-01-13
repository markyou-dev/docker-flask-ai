FROM python:3.9-slim

WORKDIR /usr/src/app

# 필수 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Flask 실행
CMD ["python", "app.py"]