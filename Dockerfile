FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from transformers import WhisperForConditionalGeneration, WhisperProcessor; WhisperForConditionalGeneration.from_pretrained('openai/whisper-small'); WhisperProcessor.from_pretrained('openai/whisper-small')"

COPY app/ ./app
COPY adapters/ ./adapters
WORKDIR /app/app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
