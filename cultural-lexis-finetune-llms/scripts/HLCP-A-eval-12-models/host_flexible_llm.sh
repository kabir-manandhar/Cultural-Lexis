#!/bin/bash

source /data/gpfs/projects/punim2219/LM_with_SWOW/sukaih/cultural-lexis-finetune-llms/utils/set_envs.sh

arg1=$1

if [ -z "$arg1" ]; then
  echo "arg1 is not set"
  exit 1
fi

get_free_gpu() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | \
  awk '{print $1}' | \
  awk '{
    if ($1 > max) { max = $1; idx = NR - 1 }
  } END { print idx }'
}

FREE_GPU=$(get_free_gpu)
export CUDA_VISIBLE_DEVICES=$FREE_GPU
echo "Using GPU: $FREE_GPU"

FREE_GPU=$(get_free_gpu)
export CUDA_VISIBLE_DEVICES=$FREE_GPU
echo "Running on GPU $FREE_GPU"

if [[ "$arg1" == "1" ]]; then   # 1 is - Qwen vanilla for zh   :8001
  python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct  \
  --tensor-parallel-size 1 \
  --distributed-executor-backend ray \
  --max-model-len 8192 \
  --port 8001 

elif [[ "$arg1" == "2" ]]; then # 2 is - Qwen vanilla for us   :8002
  python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct  \
  --tensor-parallel-size 1 \
  --distributed-executor-backend ray \
  --max-model-len 8192 \
  --port 8002
elif [[ "$arg1" == "3" ]]; then # 3 is - Qwen SWOW zh          :8003
  python -m vllm.entrypoints.openai.api_server \
  --model /data/projects/punim0478/sukaih/Sukai_Project/huahua/data/07_model_output/qwen/swow_zh/lora_combined  \
  --tensor-parallel-size 1 \
  --distributed-executor-backend ray \
  --max-model-len 8192 \
  --port 8003 \
  --served-model-name sukai/qwen_swow_zh 
elif [[ "$arg1" == "4" ]]; then # 4 is - Qwen SWOW us          :8004
  python -m vllm.entrypoints.openai.api_server \
  --model /data/projects/punim0478/sukaih/Sukai_Project/huahua/data/07_model_output/qwen/swow_us/lora_combined  \
  --tensor-parallel-size 1 \
  --distributed-executor-backend ray \
  --max-model-len 8192 \
  --port 8004 \
  --served-model-name sukai/qwen_swow_us
elif [[ "$arg1" == "5" ]]; then # 5 is - Qwen PPO zh           :8005
  python -m vllm.entrypoints.openai.api_server \
  --model /data/projects/punim0478/sukaih/Sukai_Project/huahua/data/07_model_output/qwen/swow_zh/pure_ppo  \
  --tensor-parallel-size 1 \
  --distributed-executor-backend ray \
  --max-model-len 8192 \
  --port 8005 \
  --served-model-name sukai/qwen_ppo_zh \
  --gpu_memory_utilization 0.46
elif [[ "$arg1" == "6" ]]; then # 6 is - Qwen PPO us           :8006
  python -m vllm.entrypoints.openai.api_server \
  --model /data/projects/punim0478/sukaih/Sukai_Project/huahua/data/07_model_output/qwen/swow_us/pure_ppo  \
  --tensor-parallel-size 1 \
  --distributed-executor-backend ray \
  --max-model-len 8192 \
  --port 8006 \
  --served-model-name sukai/qwen_ppo_us \
  --gpu_memory_utilization 0.46
elif [[ "$arg1" == "7" ]]; then # 7 is - Llama vanilla for zh  :8007
  python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct  \
  --tensor-parallel-size 1 \
  --distributed-executor-backend ray \
  --max-model-len 8192 \
  --port 8007 \
  --gpu_memory_utilization 0.46
elif [[ "$arg1" == "8" ]]; then # 8 is - Llama vanilla for us  :8008
  python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct  \
  --tensor-parallel-size 1 \
  --distributed-executor-backend ray \
  --max-model-len 8192 \
  --port 8008 \
  --gpu_memory_utilization 0.46
elif [[ "$arg1" == "9" ]]; then # 9 is - Llama SWOW zh         :8009
  python -m vllm.entrypoints.openai.api_server \
  --model /data/projects/punim0478/sukaih/Sukai_Project/huahua/data/07_model_output/llama3/swow_zh/lora_combined  \
  --tensor-parallel-size 1 \
  --distributed-executor-backend ray \
  --max-model-len 8192 \
  --port 8009 \
  --served-model-name sukai/llama_swow_zh \
  --gpu_memory_utilization 0.46
elif [[ "$arg1" == "10" ]]; then # 10 is - Llama SWOW us         :8010
  python -m vllm.entrypoints.openai.api_server \
  --model /data/projects/punim0478/sukaih/Sukai_Project/huahua/data/07_model_output/llama3/swow_us/lora_combined  \
  --tensor-parallel-size 1 \
  --distributed-executor-backend ray \
  --max-model-len 8192 \
  --port 8010 \
  --served-model-name sukai/llama_swow_us \
  --gpu_memory_utilization 0.46
elif [[ "$arg1" == "11" ]]; then # 11 is - Llama PPO zh          :8011
  python -m vllm.entrypoints.openai.api_server \
  --model /data/projects/punim0478/sukaih/Sukai_Project/huahua/data/07_model_output/llama3/swow_zh/pure_ppo  \
  --tensor-parallel-size 1 \
  --distributed-executor-backend ray \
  --max-model-len 8192 \
  --port 8011 \
  --served-model-name sukai/llama_ppo_zh \
  --gpu_memory_utilization 0.46
elif [[ "$arg1" == "12" ]]; then # 12 is - Llama PPO us          :8012
  python -m vllm.entrypoints.openai.api_server \
  --model /data/projects/punim0478/sukaih/Sukai_Project/huahua/data/07_model_output/llama3/swow_us/pure_ppo  \
  --tensor-parallel-size 1 \
  --distributed-executor-backend ray \
  --max-model-len 8192 \
  --port 8012 \
  --served-model-name sukai/llama_ppo_us \
  --gpu_memory_utilization 0.46


else
  echo "Invalid argument. Please provide a number between 1 and 12."
  exit 1
fi