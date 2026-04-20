vllm serve Qwen/Qwen3-32B \
  --enable-reasoning \
  --reasoning-parser deepseek_r1 \
  --tensor-parallel-size 4 \
  --distributed-executor-backend ray \
  --max-model-len 8192


