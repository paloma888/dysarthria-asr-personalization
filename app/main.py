from fastapi import FastAPI, UploadFile, File
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
import torchaudio
import io
import time

POPULATION_ADAPTER_PATH = "../adapters/adapter-dropout"
PERSONAL_ADAPTER_PATH = "../adapters/adapter-personalize-demo"
MODEL_PATH = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(MODEL_PATH)
device = "cpu"

model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model = PeftModel.from_pretrained(model, POPULATION_ADAPTER_PATH).merge_and_unload()
model = PeftModel.from_pretrained(model, PERSONAL_ADAPTER_PATH)
model = model.merge_and_unload()


app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File()):
    contents = await file.read()

    start = time.perf_counter()
    waveform, sampling_rate = torchaudio.load(io.BytesIO(contents))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    if sampling_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sampling_rate, 16000)

    waveform = waveform.squeeze()
    waveform = waveform.numpy()
    input_features = processor.feature_extractor(waveform, sampling_rate=16000, return_tensors="pt").input_features.to(device)

    tokens = model.generate(input_features, language="en", task="transcribe", max_new_tokens=64)
    predicted_text = processor.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
    end = time.perf_counter()
    elapsed = (end - start) * 1000

    return {"text": predicted_text, "latency_ms": round(elapsed, 1)}

