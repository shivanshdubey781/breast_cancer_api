# Dockerfile

FROM python:3.9

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "fastapi_breast_cancer_api:app", "--host", "0.0.0.0", "--port", "8000"]
