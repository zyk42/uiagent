#!/bin/bash
# 量化: awq-w4a16  (GUI-Owl-1.5-32B-Instruct)
# 用法: bash run_awq-w4a16.sh    (建议: nohup bash run_awq-w4a16.sh > logs/awq-w4a16.out 2>&1 &)
set -e
cd /root/quant_calib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
mkdir -p logs
python run_quant.py awq-w4a16 2>&1 | tee logs/awq-w4a16.log
