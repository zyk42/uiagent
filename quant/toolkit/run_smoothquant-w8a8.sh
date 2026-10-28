#!/bin/bash
# 量化: smoothquant-w8a8  (GUI-Owl-1.5-32B-Instruct)
# 用法: bash run_smoothquant-w8a8.sh    (建议: nohup bash run_smoothquant-w8a8.sh > logs/smoothquant-w8a8.out 2>&1 &)
set -e
cd /root/quant_calib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
mkdir -p logs
python run_quant.py smoothquant-w8a8 2>&1 | tee logs/smoothquant-w8a8.log
