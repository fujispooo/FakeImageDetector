FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

COPY src/ src/
COPY README.md .

RUN pip install --no-cache-dir -e .

EXPOSE 8000

ENTRYPOINT ["fake-image-detector"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
