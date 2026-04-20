#!/bin/bash

source env.sh

kedro run -p ppo_mcq_data_generation --params "ppo_mcq_params.model_params.model_type=llama3,ppo_mcq_params.model_params.device=cuda:0"

# kedro run -p ppo_mcq_data_generation --params ppo_mcq_params.model_params.model_type=qwen