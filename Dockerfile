# Prosper scheduler: voice bot + builder UI + evaluation tools, in one image.
#
#   docker compose up --build        # then open http://localhost:7860/builder
#
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# OpenCV (a pipecat dependency) needs these shared libraries even when unused.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend

COPY backend/requirements.txt .
RUN pip install -r requirements.txt

# Pipecat's text aggregation downloads NLTK data on first import; bake it in so the
# container starts offline-safe and fast.
RUN python -c "import nltk; nltk.download('punkt_tab', quiet=True)"

COPY backend/ .

EXPOSE 7860
CMD ["python", "bot.py", "--host", "0.0.0.0", "--port", "7860"]
