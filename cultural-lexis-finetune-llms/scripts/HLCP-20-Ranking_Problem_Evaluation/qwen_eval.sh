#!/bin/bash

source env.sh
set -x 

CUDA_VISIBLE_DEVICES=3

kedro run -p ranking_problem_evaluation --params=ranking_problem_eval_params.model_type=qwen