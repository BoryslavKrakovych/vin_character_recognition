FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install tensorflow opencv-python numpy

RUN pip install Pillow

WORKDIR /app

COPY . /app

CMD ["python3", "/app/inference.py"]