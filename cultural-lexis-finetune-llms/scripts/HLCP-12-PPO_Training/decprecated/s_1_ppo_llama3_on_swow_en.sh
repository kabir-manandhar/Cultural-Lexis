#!/bin/bash

export CUDA_HOME=$CONDA_PREFIX
export PYTHONPATH=$PWD
export WORKING_DIR=$PWD
export HF_HOME=/data/projects/punim0478/sukaih/huggingface
export TRITON_CACHE_DIR=/data/projects/punim0478/sukaih/triton_cache
export CUDA_VISIBLE_DEVICES=0,1,2,3 # ! Change this to 0,1,2,3 if you have 4 GPUs
export NCCL_SOCKET_IFNAME=bond0 # ! very important for deepspeed multi node
export NCCL_IB_DISABLE=1 # ! very important for deepspeed multi node

# -- Vars --
export LF_TEMPLATE='llama3'
export LF_SUBSET='swow_en'
export LF_SPLIT='trl'
export LF_DATASET_DIR='scripts/HLCP-8-SFT_Training/configs/dataset_info.json'
export WANDB_PROJECT='llamafactory'
export TRAIN_IP='172.26.110.35'


if [ $LF_TEMPLATE == 'llama3' ]; then
    export THE_MODEL_NAME='meta-llama/Meta-Llama-3.1-8B-Instruct'
elif [ $LF_TEMPLATE == 'qwen' ]; then
    export THE_MODEL_NAME='Qwen/Qwen2.5-7B-Instruct'
fi

MY_MACHINE_RANK=$1 # total we have 2 machines
MY_PROCESS_PORT='25588'


accelerate launch \
    --config_file scripts/HLCP-11-PPO_Training/configs/deepspeed_zero2.yaml \
    --main_process_ip $TRAIN_IP \
    --main_process_port $MY_PROCESS_PORT \
    --machine_rank $MY_MACHINE_RANK \
    src/cultural_lexis_finetune_llms/pipelines/ppo_further_training/ppo_train.py \
    --output_dir "/data/projects/punim0478/sukaih/Sukai_Project/huahua/data/07_model_output/${LF_TEMPLATE}/${LF_SUBSET}/ppo" \
    --dataset_name llm_swow_finetune_dataset \
    --num_ppo_epochs 4 \
    --num_mini_batches 4 \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 3 \
    --gradient_accumulation_steps 3 \
    --total_episodes 1000000 \
    --model_name_or_path $THE_MODEL_NAME \
    --sft_model_path "/data/projects/punim0478/sukaih/Sukai_Project/huahua/data/07_model_output/${LF_TEMPLATE}/${LF_SUBSET}/sft" \
    --local_rollout_forward_batch_size 2 \
    --stop_token eos \
    --report_to wandb \
    --run_name s_1_ppo_llama3_on_swow_en 
    
    