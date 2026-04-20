python -m vllm.entrypoints.openai.api_server \
  --model /data/projects/punim0478/sukaih/Sukai_Project/huahua/data/07_model_output/qwen/swow_zh/lora_combined  \
  --tensor-parallel-size 2 \
  --distributed-executor-backend ray \
  --max-model-len 8192 \
  --port 8001 \
  --served-model-name sukai/self_model


