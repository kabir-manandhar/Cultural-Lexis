#!/bin/bash

source env.sh

set -x 

working_dir=/data/gpfs/projects/punim2219/LM_with_SWOW/sukaih/cultural-lexis-finetune-llms
pretrain_model_dir=/data/projects/punim0478/sukaih/Sukai_Project/huahua/data/07_model_output/qwen/swow_en
output_tag=ftd_n_ppo
pretrain_model=/data/projects/punim0478/sukaih/Sukai_Project/huahua/data/07_model_output/qwen/swow_en/lora_combined
data_info=/data/03_primary/openrlhf_dataset/swow_en
# ! also need to change env_vars
source scripts/HLCP-12-PPO_Training/reuse_configs.sh

# ! actor_num_gpus_per_node * actor_num_nodes should >= vllm_num_engines

# vllm_num_engines should be equal to number of node 
# vllm_tensor_parallel_size should be equal to number of gpus per node
# ray job delete/stop JOB_ID http://127.0.0.1:8265

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/data/gpfs/projects/punim2219/LM_with_SWOW/sukaih/cultural-lexis-finetune-llms", "env_vars": {"DATA_NAME": "swow_en"}}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   ${PPO_FLAGS} \
   --use_wandb e306c9c2aee5b8224dcfeaab47393338739db3fe \
   --wandb_project llamafactory \
   --wandb_run_name ${output_tag}${pretrain_model}_swow_en