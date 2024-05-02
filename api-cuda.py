from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
model_ckpt = "models/xlm-roberta-base-language-detection"
pipe = pipeline("text-classification", model=model_ckpt, device="cuda:0")


class Input(BaseModel):
    text: str


@app.post("/process")
async def process(input: Input) -> str:
    return pipe(input.text, top_k=1, truncation=True)[0]['label']
