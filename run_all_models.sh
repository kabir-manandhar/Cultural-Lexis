

############################################################################################
# ###################################  Chinese  + Llama3.1 ###################################
# # 1. baseline model 
# python language_modelling/main.py \
# --country_name  "China" --model_path /data/gpfs/projects/punim2219/chunhua/cache_dir/Meta-Llama-3.1-8B-Instruct/ \
# --output_file /data/gpfs/projects/punim2219/LM_with_SWOW/chunhua/Data/output/llama/WVS_ZH_llama_vanilla.json

# # 2. SWOW.en -> SFT Llama3.1 
# python language_modelling/main.py \
# --country_name  "China" \
# --model_path /data/projects/punim0478/sukaih/Sukai_Project/huahua/data/07_model_output/llama3/swow_zh/lora_combined/ \
# --output_file /data/gpfs/projects/punim2219/LM_with_SWOW/chunhua/Data/output/llama/WVS_ZH_llama_sft.json 

# # 3. SWOW.en ->  PPO Llama3.1 
# python language_modelling/main.py \
# --country_name  "China" \
# --model_path /data/projects/punim0478/sukaih/Sukai_Project/huahua/data/07_model_output/llama3/swow_zh/pure_ppo \
# --output_file  /data/gpfs/projects/punim2219/LM_with_SWOW/chunhua/Data/output/llama/WVS_ZH_llama_ppo.json

# # 4. SWOW.en -> SFT + PPO Llama3.1 
# python language_modelling/main.py \
# --country_name  "China" \
# --model_path /data/projects/punim0478/sukaih/Sukai_Project/huahua/data/07_model_output/llama3/swow_zh/ftd_n_ppo \
# --output_file /data/gpfs/projects/punim2219/LM_with_SWOW/chunhua/Data/output/llama/WVS_ZH_llama_ftd_n_ppo.json

# ############################################################################################






############################################################################################
###################################  Chinese  + QWEN ###################################
# # 1. baseline model 
# python language_modelling/main.py \
# --country_name  "China" \
# --model_path /data/gpfs/projects/punim2219/chunhua/cache_dir/Qwen2.5-7B-Instruct \
# --output_file /data/gpfs/projects/punim2219/LM_with_SWOW/chunhua/Data/output/qwen/WVS_ZH_qwen_vanilla.json

# # 2. SWOW.en ->  SFT 
# python language_modelling/main.py \
# --country_name  "China" \
# --model_path /data/projects/punim0478/sukaih/Sukai_Project/huahua/data/07_model_output/qwen/swow_zh/lora_combined \
# --output_file /data/gpfs/projects/punim2219/LM_with_SWOW/chunhua/Data/output/qwen/WVS_ZH_qwen_sft.json

# # 3. SWOW.en ->  PPO 
# python language_modelling/main.py \
# --country_name  "China" \
# --model_path /data/projects/punim0478/sukaih/Sukai_Project/huahua/data/07_model_output/qwen/swow_zh/pure_ppo \
# --output_file /data/gpfs/projects/punim2219/LM_with_SWOW/chunhua/Data/output/qwen/WVS_ZH_qwen_ppo.json 


# # 4. SWOW.en -> SFT + PPO 
# python language_modelling/main.py \
# --country_name  "China" \
# --model_path /data/projects/punim0478/sukaih/Sukai_Project/huahua/data/07_model_output/qwen/swow_zh/ftd_n_ppo \
# --output_file /data/gpfs/projects/punim2219/LM_with_SWOW/chunhua/Data/output/qwen/WVS_ZH_qwen_ftd_n_ppo.json
############################################################################################


# ############################################################################################
# ###################################  English  + Llama3.1 ###################################
# # 1. baseline model 
# python language_modelling/main.py \
# --country_name  "United States" \
# --model_path /data/gpfs/projects/punim2219/chunhua/cache_dir/Meta-Llama-3.1-8B-Instruct/ \
# --output_file /data/gpfs/projects/punim2219/LM_with_SWOW/chunhua/Data/output/llama/WVS_US_llama_vanilla.json

# # 2. SWOW.en -> SFT Llama3.1 
# python language_modelling/main.py \
# --country_name  "United States" \
# --model_path /data/projects/punim0478/sukaih/Sukai_Project/huahua/data/07_model_output/llama3/swow_en/lora_combined/ \
# --output_file /data/gpfs/projects/punim2219/LM_with_SWOW/chunhua/Data/output/llama/WVS_US_llama_sft.json

# # 3. SWOW.en ->  PPO Llama3.1 
python language_modelling/main.py \
--country_name  "United States" \
--model_path /data/projects/punim0478/sukaih/Sukai_Project/huahua/data/07_model_output/llama3/swow_en/pure_ppo \
--output_file /data/gpfs/projects/punim2219/LM_with_SWOW/chunhua/Data/output/llama/WVS_US_llama_ppo.json


# # 4. SWOW.en -> SFT + PPO Llama3.1 
# python language_modelling/main.py \
# --country_name  "United States" \
# --model_path /data/projects/punim0478/sukaih/Sukai_Project/huahua/data/07_model_output/llama3/swow_en/ftd_n_ppo \
# --output_file /data/gpfs/projects/punim2219/LM_with_SWOW/chunhua/Data/output/llama/WVS_US_llama_ftd_n_ppo.json
# ############################################################################################


# ############################################################################################
# ###################################  English  + QWEN ###################################
# # 1. baseline model 
# python language_modelling/main.py \
# --country_name  "United States" \
# --model_path /data/gpfs/projects/punim2219/chunhua/cache_dir/Qwen2.5-7B-Instruct \
# --output_file /data/gpfs/projects/punim2219/LM_with_SWOW/chunhua/Data/output/qwen/WVS_US_qwen_vanilla.json


# # 2. SWOW.en ->  SFT 
# # echo "2. SWOW.en -> SFT"
# python language_modelling/main.py \
# --country_name  "United States" \
# --model_path /data/projects/punim0478/sukaih/Sukai_Project/huahua/data/07_model_output/qwen/swow_en/lora_combined \
# --output_file /data/gpfs/projects/punim2219/LM_with_SWOW/chunhua/Data/output/qwen/WVS_US_qwen_sft.json

# 3. SWOW.en ->  PPO 
python language_modelling/main.py \
--country_name  "United States" \
--model_path /data/projects/punim0478/sukaih/Sukai_Project/huahua/data/07_model_output/qwen/swow_en/pure_ppo \
--output_file /data/gpfs/projects/punim2219/LM_with_SWOW/chunhua/Data/output/qwen/WVS_US_qwen_ppo.json

# # 4. SWOW.en -> SFT + PPO 
# python language_modelling/main.py \
# --country_name  "United States" \
# --model_path /data/projects/punim0478/sukaih/Sukai_Project/huahua/data/07_model_output/qwen/swow_en/ftd_n_ppo \
# --output_file /data/gpfs/projects/punim2219/LM_with_SWOW/chunhua/Data/output/qwen/WVS_US_qwen_ftd_n_ppo.json
# ############################################################################################

