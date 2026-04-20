#!/bin/bash

# export CUDA_HOME=$CONDA_PREFIX
# export PYTHONPATH=$PWD
# export WORKING_DIR=$PWD
# export HF_HOME=/data/projects/punim0478/sukaih/huggingface
# export TRITON_CACHE_DIR=/data/projects/punim0478/sukaih/triton_cache
# export CUDA_VISIBLE_DEVICES=0,1,2,3

source env.sh
llamafactory-cli train scripts/HLCP-8-SFT_Training/configs/s_3_sft_llama3_on_swow_zh_lora.yaml


