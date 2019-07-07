#!/bin/bash
# Name of the model
MODEL_NAME="test01"

# Prediction file을 독립적인 Bundle 파일로 바꾸기
cl make run-predictions/predictions.json -n predictions-{$MODEL_NAME}