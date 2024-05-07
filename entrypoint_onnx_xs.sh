#!/bin/bash

tar -xvf models/xlm-roberta-base-language-detection-onnx.tar.gz -C models/

uvicorn api:app --host 0.0.0.0
