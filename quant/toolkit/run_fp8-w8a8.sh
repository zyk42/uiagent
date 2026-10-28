#!/bin/bash
# 量化: fp8-w8a8 (FP8_DYNAMIC, datafree)
set -e
cd /root/quant_calib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
mkdir -p logs
python run_quant.py fp8-w8a8 2>&1 | tee logs/fp8-w8a8.log
