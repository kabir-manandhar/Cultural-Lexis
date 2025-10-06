#!/bin/bash
my_ip=`ip addr show bond0.3027 | awk '/inet / {print $2}' | cut -d'/' -f1`
echo "My IP is $my_ip"


bash scripts/HLCP-12-PPO_Training/run_cluster.sh \
    vllm/vllm-openai \
    172.26.93.168 \
    --worker \
    /data/projects/punim0478/sukaih/huggingface \
    -e VLLM_HOST_IP=${my_ip}
