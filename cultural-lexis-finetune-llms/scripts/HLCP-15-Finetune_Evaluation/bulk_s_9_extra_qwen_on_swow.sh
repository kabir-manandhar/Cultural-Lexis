#!/bin/bash

source utils/set_envs.sh
export WANDB_PROJECT='llamafactory'
CUDA_VISIBLE_DEVICES=3

# DISABLE_VERSION_CHECK=1 kedro run --pipeline=finetuning_evaluation --params=\
# finetune_eval_params.top_k=10,\
# finetune_eval_params.dataset_name=swow_en,\
# finetune_eval_params.model_type=qwen,\
# finetune_eval_params.want_few_shot_example=true

# sleep 20 

WANDB_CACHE_DIR=/data/projects/punim0478/sukaih/Sukai_Project/huahua/wandb DISABLE_VERSION_CHECK=1 kedro run --pipeline=finetuning_evaluation --params=\
finetune_eval_params.top_k=10,\
finetune_eval_params.dataset_name=swow_zh,\
finetune_eval_params.model_type=qwen,\
finetune_eval_params.want_few_shot_example=true
