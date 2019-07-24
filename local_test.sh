#!/bin/bash
EPOCH_NUM=$1
OUTPUT_FILE="predictions_epoch_$EPOCH_NUM.json"
# python3 predict_server.py <(cat mrqa-dev-data/*.jsonl) $OUTPUT_FILE 8888
for data_file in `ls mrqa-dev-data/*.jsonl`; do echo ${data_file};base=$(echo ${data_file} | cut -d "/" -f2);python3 mrqa_official_eval.py ${data_file} $OUTPUT_FILE > eval_${base::-1}; done