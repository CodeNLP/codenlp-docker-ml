from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import onnxruntime as rt
from transformers import AutoTokenizer, PretrainedConfig

app = FastAPI()

model_path = "models/xlm-roberta-base-language-detection-onnx"
tokenizer = AutoTokenizer.from_pretrained(model_path)
config = PretrainedConfig.from_pretrained(model_path)
ort_sess = rt.InferenceSession(Path(model_path) / "model_quantized.onnx")


class Input(BaseModel):
    text: str


@app.post("/process")
async def process(input: Input) -> str:
    text = [input.text]
    vector = tokenizer(text, padding=True)
    vector = {k: v for k, v in vector.items()}
    outputs = ort_sess.run(None, vector)
    label_ids = np.argmax(outputs, axis=2)
    labels = [config.id2label[label_id] for label_id in label_ids[0]]
    return labels[0]
