FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-scoreboard.txt .
# Install exactly one ONNX runtime. CUDA libraries are loaded by runtime.py.
RUN sed '/^onnxruntime$/d' requirements-scoreboard.txt > /tmp/requirements.txt && \
    pip install --no-cache-dir -r /tmp/requirements.txt "onnxruntime-gpu[cuda,cudnn]==1.21.1" "nvidia-cublas-cu12>=12,<13"
COPY studio/__init__.py studio/__init__.py
COPY studio/scoreboard studio/scoreboard
RUN python -m studio.scoreboard.fetch_models
RUN python -c "from studio.scoreboard.auto_layout import download_model; download_model()"
ENV PORT=8080 SCORESIGHT_DEVICE=cuda PYTHONUNBUFFERED=1
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "1", "--timeout", "3500", "studio.scoreboard.service:app"]
