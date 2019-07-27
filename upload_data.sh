#!/bin/bash
# File Name, settings etc.
WORKSHEET_NAME="mrqa-bert-large-adv"

####################### Basic preparation #######################
# 1. Switch the worksheet
cl work $WORKSHEET_NAME

# 2. Upload base file from mrqa 2019 (Ignore the error message)
cl add bundle mrqa2019-utils//mrqa-dev-data .
cl add bundle mrqa2019-utils//predict_server.py .
cl add bundle mrqa2019-utils//mrqa_official_eval.py .

# 3. Upload src & config folder from local
cl upload ./src
cl upload ./config  # This parts will take quite a long time (over 400MB)