#!/bin/bash

# export CUDA_HOME=$CONDA_PREFIX
# export PYTHONPATH=$PWD
# export WORKING_DIR=$PWD
# export HF_HOME=/data/projects/punim0478/sukaih/huggingface
# export TRITON_CACHE_DIR=/data/projects/punim0478/sukaih/triton_cache
# export CUDA_VISIBLE_DEVICES=0,1,2,3


# node_rank=$1
# nnodes=3
# master_addr='172.26.110.9'

# FORCE_TORCHRUN=1 NNODES=$nnodes NODE_RANK=$node_rank MASTER_ADDR=$master_addr MASTER_PORT=29500 llamafactory-cli train scripts/HLCP-8-SFT_Training/configs/s_1_sft_llama3_on_swow_en.yaml

source env.sh

# clean up cache 
# rm -rf /data/projects/punim0478/sukaih/huggingface/datasets/participant_swow_collection_us

llamafactory-cli train scripts/HLCP-8-SFT_Training/configs/s_13_sft_qwen_on_mimic_swow_zh_lora_3_epoch.yaml

sleep 10

llamafactory-cli train scripts/HLCP-8-SFT_Training/configs/s_13_sft_qwen_on_mimic_swow_rp_lora_3_epoch.yaml

sleep 10

llamafactory-cli train scripts/HLCP-8-SFT_Training/configs/s_13_sft_qwen_on_mimic_swow_nl_lora_3_epoch.yaml

sleep 10

llamafactory-cli train scripts/HLCP-8-SFT_Training/configs/s_13_sft_gptoss_on_mimic_swow_zh_lora_3_epoch.yaml

sleep 10

llamafactory-cli train scripts/HLCP-8-SFT_Training/configs/s_13_sft_gptoss_on_mimic_swow_rp_lora_3_epoch.yaml

sleep 10

llamafactory-cli train scripts/HLCP-8-SFT_Training/configs/s_13_sft_gptoss_on_mimic_swow_nl_lora_3_epoch.yaml

sleep 10

llamafactory-cli train scripts/HLCP-8-SFT_Training/configs/s_13_sft_llama3_on_mimic_swow_zh_lora_3_epoch.yaml

sleep 10

llamafactory-cli train scripts/HLCP-8-SFT_Training/configs/s_13_sft_llama3_on_mimic_swow_rp_lora_3_epoch.yaml

sleep 10

llamafactory-cli train scripts/HLCP-8-SFT_Training/configs/s_13_sft_llama3_on_mimic_swow_nl_lora_3_epoch.yaml
