# BugPredictor with the Python engine:  docker build -t bugpredictor . && docker run -p 8000:8000 bugpredictor
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir numpy scikit-learn fastapi "uvicorn[standard]" python-multipart
COPY bugpredictor bugpredictor
COPY app app
COPY data data
COPY web web
EXPOSE 8000
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
