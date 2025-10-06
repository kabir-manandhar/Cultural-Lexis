#!/bin/bash

source env.sh

set -x 

working_dir=/data/gpfs/projects/punim2219/LM_with_SWOW/sukaih/cultural-lexis-finetune-llms
pretrain_model_dir=/data/projects/punim0478/sukaih/Sukai_Project/huahua/data/07_model_output/llama3/swow_en
output_tag=pure_ppo
pretrain_model=meta-llama/Meta-Llama-3.1-8B-Instruct
data_info=/data/03_primary/openrlhf_dataset/swow_en
# ! also need to change env_vars

# ! actor_num_gpus_per_node * actor_num_nodes should >= vllm_num_engines

# vllm_num_engines should be equal to number of node 
# vllm_tensor_parallel_size should be equal to number of gpus per node
# ray job delete/stop JOB_ID http://127.0.0.1:8265

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/data/gpfs/projects/punim2219/LM_with_SWOW/sukaih/cultural-lexis-finetune-llms", "env_vars": {"DATA_NAME": "swow_en"}}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 4 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 4 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 4 \
   --lora_rank 64 \
   --lora_alpha 256 \
   --lora_dropout 0.05 \
   --vllm_num_engines 0 \
   --colocate_actor_ref \
   --pretrain ${pretrain_model} \
   --remote_rm_url ${working_dir}/src/cultural_lexis_finetune_llms/pipelines/ppo_further_training/reward_func.py \
   --save_path ${pretrain_model_dir}/${output_tag} \
   --micro_train_batch_size 8 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 16 \
   --rollout_batch_size 1024 \
   --max_samples 100000 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --generate_max_len 1024 \
   --zero_stage 2 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data ${working_dir}${data_info} \
   --input_key input \
   --label_key label \
   --apply_chat_template \
   --normalize_reward \
   --flash_attn \
   --vllm_sync_backend nccl \
   --gradient_checkpointing \
   --use_wandb e306c9c2aee5b8224dcfeaab47393338739db3fe \
   --wandb_project llamafactory \
   --wandb_run_name ${output_tag}${pretrain_model}_swow_en


   # --packing_samples \
   # --vllm_num_engines 4 \
   # --vllm_tensor_parallel_size 2 \