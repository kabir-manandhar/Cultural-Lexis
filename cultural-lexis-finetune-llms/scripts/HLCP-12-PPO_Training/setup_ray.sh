#!/bin/bash
# ssh forward port 

# ssh -fNT -L 8265:localhost:8265 spartan-gpgpu168 ;run this on the spartain entry machine
# ssh -fNT -L 8265:localhost:8265 unimeb  ; run this on your local machine and unimeb is the nickname of the remote spartan entry machine

# ssh -fNT -L 8000:localhost:8000 spartan-gpgpu135

# ssh -fNT -L 9999:localhost:9999 spartan-gpgpu135
# ssh -fNT -L 9999:localhost:9999 unimeb

# ! 135 device connect to 169 
# ssh -fNT -L 8001:localhost:8001 spartan-gpgpu169
# ssh -fNT -L 8002:localhost:8002 spartan-gpgpu169

# ssh -fNT -L 8005:localhost:8005 spartan-gpgpu070
# ssh -fNT -L 8011:localhost:8011 spartan-gpgpu070

# http://127.0.0.1:9999/tree?token=8233eba6014cc2a47180d6037e4adea0f7f2d71fab167181

# if want to remove the port forwarding
# ps aux | grep localhost:8265
# kill -9 {PID}
# ps aux | grep localhost:8265 | awk {'print $2'} | xargs kill -9


# if you want to launch ray on more nodes, use
# ray start --address {MASTER-NODE-ADDRESS}:6379  --num-gpus 4


bash scripts/HLCP-12-PPO_Training/run_cluster.sh \
        vllm/vllm-openai \
        172.26.93.141 \
        --head \
        /data/projects/punim0478/sukaih/huggingface \
        -e VLLM_HOST_IP=172.26.93.141
