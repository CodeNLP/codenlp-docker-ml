import json
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import onnxruntime as rt
from tokenizers import Tokenizer

app = FastAPI()

model_path = "models/xlm-roberta-base-language-detection-onnx"
tokenizer = Tokenizer.from_file(model_path + "/tokenizer.json")
with open(model_path + "/config.json", "r", encoding="utf-8") as fin:
    config = json.load(fin)
ort_sess = rt.InferenceSession(Path(model_path) / "model_quantized.onnx")


class Input(BaseModel):
    text: str


@app.post("/process")
async def process(input: Input) -> str:
    encoded = tokenizer.encode(input.text)
    vector = {
        "input_ids": [encoded.ids],
        "attention_mask": [encoded.attention_mask],
    }
    outputs = ort_sess.run(None, vector)
    label_ids = np.argmax(outputs, axis=2)
    labels = [config["id2label"][str(label_id)] for label_id in label_ids[0]]
    return labels[0]
