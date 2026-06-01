FROM python:3.12-slim

WORKDIR /app

# Copy semua file dari MLProject
COPY . .

# Install dependencies
RUN pip install --no-cache-dir \
    mlflow \
    pandas \
    numpy \
    scikit-learn \
    matplotlib

# Port untuk MLflow model serving
EXPOSE 5000

# Default: jalanin training dulu
CMD ["python", "modelling.py"]
