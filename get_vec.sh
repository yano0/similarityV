#!/bin/sh
device=cuda:0
model=cl-tohoku/bert-base-japanese-whole-word-masking
input_path=data/tweet.jsonl

python main.py \
    --device ${device} \
    --model ${model} \
    # --input_path ${input_path}