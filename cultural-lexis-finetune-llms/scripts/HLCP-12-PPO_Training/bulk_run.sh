#!/bin/bash

# check whether PID is alive, loop until it is dead, use ps aux | grep bash to find the PID

# while kill -0 45456; do
#     echo "Process is still alive"
#     echo "time: $(date)"
#   sleep 320
# done

# sleep 20


# bash scripts/HLCP-12-PPO_Training/s1_ppo_llama_swow_en_single.sh

# sleep 20

# bash scripts/HLCP-12-PPO_Training/s2_ppo_llama_swow_zh_single.sh

# sleep 20

# bash scripts/HLCP-12-PPO_Training/s3_ppo_qwen_swow_zh_single.sh

# sleep 20

bash scripts/HLCP-12-PPO_Training/s4_ppo_qwen_swow_en_single.sh

# sleep 20

# bash scripts/HLCP-12-PPO_Training/s5_ppo_llama_swow_en_fted.sh

# sleep 20

# bash scripts/HLCP-12-PPO_Training/s6_ppo_llama_swow_zh_fted.sh

# sleep 20

# bash scripts/HLCP-12-PPO_Training/s7_ppo_qwen_swow_zh_fted.sh

# sleep 20

# bash scripts/HLCP-12-PPO_Training/s8_ppo_qwen_swow_en_fted.sh
