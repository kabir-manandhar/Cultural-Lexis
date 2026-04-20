export CUDA_HOME=$CONDA_PREFIX
export PYTHONPATH=$PWD
export WORKING_DIR=$PWD
export HF_HOME=/data/projects/punim0478/sukaih/huggingface
export TRITON_CACHE_DIR=/data/projects/punim0478/sukaih/triton_cache
export NCCL_SOCKET_IFNAME=bond0.3027 # ! very important for deepspeed multi node
export NCCL_IB_DISABLE=1 # ! very important for deepspeed multi node


set -x 

working_dir=/data/gpfs/projects/punim2219/LM_with_SWOW/sukaih/cultural-lexis-finetune-llms

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/data/gpfs/projects/punim2219/LM_with_SWOW/sukaih/cultural-lexis-finetune-llms", "env_vars": {"DATA_NAME": "swow_en"}}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 4 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 4 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 4 \
   --vllm_num_engines 0 \
   --colocate_actor_ref \
   --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
   --remote_rm_url ${working_dir}/src/cultural_lexis_finetune_llms/pipelines/ppo_further_training/reward_func.py \
   --save_path /data/projects/punim0478/sukaih/examples/checkpoint/llama3-8b-rlhf \
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
   --prompt_data ${working_dir}/data/03_primary/openrlhf_dataset/swow_en \
   --input_key input \
   --label_key label \
   --apply_chat_template \
   --normalize_reward \
   --flash_attn \
   --gradient_checkpointing \
   --use_wandb e306c9c2aee5b8224dcfeaab47393338739db3fe