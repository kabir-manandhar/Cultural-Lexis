#!/bin/bash

conda install -c conda-forge cudatoolkit-dev

conda install pytorch=2.4.0 torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

pip install https://github.com/hiyouga/LLaMA-Factory/archive/refs/heads/main.tar.gz#egg=llamafactory[torch,metrics,deepspeed]

pip install -r requirements.txt

pip install flash-attn==2.6.3 

# the following should be done after sft training 
pip uninstall deepspeed accelerate datasets transformers trl bitsandbytes
pip install git+https://github.com/huggingface/trl.git#egg=trl[deepspeed,quantization]
pip install 'accelerate<1.0.0'
pip install 'transformers~=4.47.1'
pip install 'deepspeed~=0.14.4'


# vim /data/gpfs/projects/punim2219/LM_with_SWOW/sukaih/lmswow/lib/python3.11/site-packages/llamafactory/extras/misc.py to modify checker 