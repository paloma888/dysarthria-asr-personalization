from fastapi import FastAPI, UploadFile, File
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
import torchaudio

ADAPTER_PATH = "../adapters/adapter-personalize-demo"
MODEL_PATH = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(MODEL_PATH)
device = "cpu"

model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model = model.merge_and_unload()


app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File()):
    contents = await file.read()
    return {"received" : len(contents)}
