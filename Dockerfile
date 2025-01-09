FROM python:3.9-slim

WORKDIR /app

# 필수 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 파일 복사
COPY ./app /app

# Flask 실행
CMD ["python", "app.py"]