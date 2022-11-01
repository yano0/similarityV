#!/bin/sh
device=cuda:0
text=おバイオ

python load.py \
    --text ${text} \
    --device ${device}