#!/bin/bash

source utils/set_envs.sh
export WANDB_PROJECT='llamafactory'
CUDA_VISIBLE_DEVICES=1

kedro run --pipeline=finetuning_evaluation --params=\
finetune_eval_params.top_k=50,\
finetune_eval_params.dataset_name=swow_en,\
finetune_eval_params.model_type=qwen


    
    