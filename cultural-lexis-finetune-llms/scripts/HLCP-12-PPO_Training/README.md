```bash
#!/bin/bash

# Define the configuration flags as a single variable
PPO_FLAGS="--ref_num_nodes 1 \
    --ref_num_gpus_per_node 4 \
    --colocate_actor_ref \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 2 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 4 \
    --packing_samples \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 2 \
    --pretrain ${pretrain_model} \
    --remote_rm_url ${working_dir}/src/cultural_lexis_finetune_llms/pipelines/ppo_further_training/reward_func.py \
    --save_path ${pretrain_model_dir}/${output_tag} \
    --micro_train_batch_size 8 \
    --train_batch_size 32 \
    --micro_rollout_batch_size 16 \
    --rollout_batch_size 64 \
    --max_samples 1000000 \
    --max_epochs 1 \
    --prompt_max_len 1024 \
    --generate_max_len 1024 \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.1 \
    --prompt_data ${working_dir}${data_info} \
    --input_key input \
    --label_key label \
    --apply_chat_template \
    --normalize_reward \
    --flash_attn \
    --adam_offload \
    --vllm_sync_backend nccl \
    --save_steps 400 \
    --save_hf_ckpt \
    --ckpt_path ${pretrain_model_dir}/ckpt/ \
    --gradient_checkpointing"
```

## Explanation of the Configs 
- ref_num_nodes: Number of nodes for the reference model.
- colocate_actor_ref: if True, the actor and the reference model will share the same node and even the same GPU. 
- critic_num_nodes: Number of nodes for the critic model.
- vllm_num_engines: how many nodes will be used for vLLM inference.
- vllm_tensor_parallel_size: how many GPUs will be used for vLLM inference per node.
- micro_train_batch_size: per GPU batch size for training.
- train_batch_size: Global training batch size. 
- micro_rollout_batch_size:  Batch size per GPU for generation
- rollout_batch_size: Replay Buffer Size is rollout_batch_size * n_samples_per_prompt
## Heuristic 
- micro_train_batch_size 8 fits for 4 GPU Actors. It crashes if only 2 GPUs are used.