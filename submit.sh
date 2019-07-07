#!/bin/bash

# 주의! 반드시 web에 가서 Description 먼저 바꾸기

# Name of the model
MODEL_NAME="test01"

# Prediction file을 독립적인 Bundle 파일로 바꾸기
cl edit predictions-{$MODEL_NAME} --tags mrqa2019-submit