#!/bin/bash
export CUDA_HOME=$CONDA_PREFIX
export PYTHONPATH=$PWD
export WORKING_DIR=$PWD
export HF_HOME=/data/projects/punim0478/sukaih/huggingface
export TRITON_CACHE_DIR=/data/projects/punim0478/sukaih/triton_cache
export CUDA_VISIBLE_DEVICES=0,1,2,3 # ! Change this to 0,1,2,3 if you have 4 GPUs
export NCCL_SOCKET_IFNAME=bond0 # ! very important for deepspeed multi node
export NCCL_IB_DISABLE=0 # ! very important for deepspeed multi node
