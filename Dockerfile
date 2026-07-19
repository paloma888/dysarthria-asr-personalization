FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir torch==2.12.1 torchaudio==2.11.0 torchcodec==0.14.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from transformers import WhisperForConditionalGeneration, WhisperProcessor; WhisperForConditionalGeneration.from_pretrained('openai/whisper-small'); WhisperProcessor.from_pretrained('openai/whisper-small')"

COPY app/ ./app
COPY adapters/ ./adapters
WORKDIR /app/app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
