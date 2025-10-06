python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct  \
  --tensor-parallel-size 2 \
  --distributed-executor-backend ray \
  --max-model-len 8192 \
  --port 8002 


