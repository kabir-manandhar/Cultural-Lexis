#!/bin/bash

# base_model=meta-llama/Meta-Llama-3.1-8B-Instruct
# swow_type=swow_en
# model_type=llama3

# python -m openrlhf.cli.lora_combiner \
#     --model_path ${base_model} \
#     --lora_path /data/projects/punim0478/sukaih/Sukai_Project/huahua/data/07_model_output/${model_type}/${swow_type}/sft_lora \
#     --output_path /data/projects/punim0478/sukaih/Sukai_Project/huahua/data/07_model_output/${model_type}/${swow_type}/lora_combined 



# base_model=meta-llama/Meta-Llama-3.1-8B-Instruct
# swow_type=swow_zh
# model_type=llama3

# python -m openrlhf.cli.lora_combiner \
#     --model_path ${base_model} \
#     --lora_path /data/projects/punim0478/sukaih/Sukai_Project/huahua/data/07_model_output/${model_type}/${swow_type}/sft_lora \
#     --output_path /data/projects/punim0478/sukaih/Sukai_Project/huahua/data/07_model_output/${model_type}/${swow_type}/lora_combined 



# base_model=Qwen/Qwen2.5-7B-Instruct
# swow_type=swow_en
# model_type=qwen

# python -m openrlhf.cli.lora_combiner \
#     --model_path ${base_model} \
#     --lora_path /data/projects/punim0478/sukaih/Sukai_Project/huahua/data/07_model_output/${model_type}/${swow_type}/sft_lora \
#     --output_path /data/projects/punim0478/sukaih/Sukai_Project/huahua/data/07_model_output/${model_type}/${swow_type}/lora_combined 


# base_model=Qwen/Qwen2.5-7B-Instruct
# swow_type=swow_zh
# model_type=qwen

# python -m openrlhf.cli.lora_combiner \
#     --model_path ${base_model} \
#     --lora_path /data/projects/punim0478/sukaih/Sukai_Project/huahua/data/07_model_output/${model_type}/${swow_type}/sft_lora \
#     --output_path /data/projects/punim0478/sukaih/Sukai_Project/huahua/data/07_model_output/${model_type}/${swow_type}/lora_combined 


base_model=Qwen/Qwen2.5-7B-Instruct
swow_type=swow_us
model_type=qwen

python -m openrlhf.cli.lora_combiner \
    --model_path ${base_model} \
    --lora_path /data/projects/punim0478/sukaih/Sukai_Project/huahua/data/07_model_output/${model_type}/${swow_type}/sft_lora \
    --output_path /data/projects/punim0478/sukaih/Sukai_Project/huahua/data/07_model_output/${model_type}/${swow_type}/lora_combined 


base_model=meta-llama/Meta-Llama-3.1-8B-Instruct
swow_type=swow_us
model_type=llama3

python -m openrlhf.cli.lora_combiner \
    --model_path ${base_model} \
    --lora_path /data/projects/punim0478/sukaih/Sukai_Project/huahua/data/07_model_output/${model_type}/${swow_type}/sft_lora \
    --output_path /data/projects/punim0478/sukaih/Sukai_Project/huahua/data/07_model_output/${model_type}/${swow_type}/lora_combined 
