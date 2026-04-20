#!/bin/bash

source utils/set_envs.sh
export WANDB_PROJECT='llamafactory'
CUDA_VISIBLE_DEVICES=0

kedro run --pipeline=finetuning_evaluation --params=\
finetune_eval_params.top_k=10,\
finetune_eval_params.dataset_name=swow_zh,\
finetune_eval_params.model_type=llama3,\
finetune_eval_params.want_few_shot_example=true


    
    