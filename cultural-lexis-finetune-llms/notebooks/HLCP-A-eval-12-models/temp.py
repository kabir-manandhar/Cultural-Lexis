# %%
import requests
import json
import os
from openai import OpenAI
from openai import AsyncOpenAI
import jsonlines
from tqdm.auto import tqdm
from IPython.display import HTML
import subprocess
from pathlib import Path
import asyncio
import time
import pandas as pd
from IPython.display import display
import asyncio


# %%
model_map = {
    1: "Qwen/Qwen2.5-7B-Instruct",
    2: "Qwen/Qwen2.5-7B-Instruct",
    3: "sukai/qwen_swow_zh",
    4: "sukai/qwen_swow_us",
    5: "sukai/qwen_ppo_zh",
    6: "sukai/qwen_ppo_us",
    7: "meta-llama/Meta-Llama-3.1-8B-Instruct",
    8: "meta-llama/Meta-Llama-3.1-8B-Instruct",
    9: "sukai/llama_swow_zh",
    10: "sukai/llama_swow_us",
    11: "sukai/llama_ppo_zh",
    12: "sukai/llama_ppo_us",
}
id_to_language_map = {
    1: 'zh',
    2: 'en',
    3: 'zh',
    4: 'en',
    5: 'zh',
    6: 'en',
    7: 'zh',
    8: 'en',
    9: 'zh',
    10: 'en',
    11: 'zh',
    12: 'en'
}

language_to_id_map = {
    'zh': [1, 3, 5, 7, 9, 11],
    'en': [2, 4, 6, 8, 10, 12]
}

# %%
# get test set data
zh_swow_test_data_fp = '/data/gpfs/projects/punim2219/LM_with_SWOW/sukaih/cultural-lexis-finetune-llms/data/03_primary/llm_swow_finetune_dataset/swow_zh/test/chunk_0.jsonl'

en_swow_test_data_fp = '/data/gpfs/projects/punim2219/LM_with_SWOW/sukaih/cultural-lexis-finetune-llms/data/03_primary/llm_swow_finetune_dataset/swow_us/test/chunk_0.jsonl'


data_dict_zh = {} # key is the cue word, vals are the list of associated words
with jsonlines.open(zh_swow_test_data_fp, 'r') as reader:
    # collect all data into a list
    for obj in tqdm(reader):
        data_dict_zh[obj['input']] = obj['output']

data_dict_en = {} # key is the cue word, vals are the list of associated words
with jsonlines.open(en_swow_test_data_fp, 'r') as reader:
    # collect all data into a list
    for obj in tqdm(reader):
        data_dict_en[obj['input']] = obj['output']

# %%
print(f"{1:02d}")

# %%
async def get_reasoning_and_content(messages, 
                             model_type:int,
                             want_print=False):
    """
    Sends chat messages to a vLLM server and returns a tuple of (reasoning_content, real_content).
    """
    reasoning_flag = False
    model=model_map[model_type]
    openai_api_key="EMPTY"
    openai_api_base=f"http://localhost:80{model_type:02d}/v1"

    client = AsyncOpenAI( # set to AsyncOpenAI for async support
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    
    params = {
        "model": model,
        "messages": messages,
        "stream": True,
    }
    if not reasoning_flag:
        params["max_tokens"] = 120
    
      
    stream = await client.chat.completions.create(
        **params
    )

    reasoning_content = ""
    real_content = ""
    start_reasoning_flg = False
    reasoning_inidicator = False
    count = 0

    async for chunk in stream:
        count += 1
        # Handle reasoning content if present
        if chunk.choices and chunk.choices[0].delta and hasattr(chunk.choices[0].delta, "reasoning_content"):
            if not start_reasoning_flg and not reasoning_inidicator:
                if want_print:
                    print("\n=== Reasoning: ===")
                reasoning_inidicator = True
            reasoning_content += chunk.choices[0].delta.reasoning_content
            if want_print:
                print(chunk.choices[0].delta.reasoning_content, end="", flush=True)
            start_reasoning_flg = True
        else:
            if count > 3 and start_reasoning_flg:
                if want_print:
                    print("\n=== End Reasoning ===")
                start_reasoning_flg = False
        # Handle main content
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            real_content += chunk.choices[0].delta.content
            if want_print:
                print(chunk.choices[0].delta.content, end="", flush=True)
    if want_print:
        print()
    return reasoning_content, real_content

# %%
# no async way time spent:  12/12 [00:17<00:00,  1.48s/it]
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "请帮我根据提示词，将相对应的关联词，关联词已经根据关联频率从高到低排好序了，请你将关联词分成四份，并进行造句"},
]

messages = [
    # {"role": "system", "content": "您是一款专为全面探索词语关联而设计的高级语言模型。"},
    {"role": "system", "content": "You are a helpful assistant."},
    
    {"role": "user", "content": "给定一个提示词，你的任务是生成一个与该提示词相关联的全面词汇列表。目标是尽可能涵盖所有相关的语境、用法和含义，避免重复相似的概念。不要生成受其他词语存在影响的词语，而是专注于提示词本身。提示词：方便。"},
]


# print("=== Testing ===")
# for model_type in tqdm([5,11]):
#     print(f"=== Model {model_type} ===")
#     output = subprocess.run(["curl", "-X", "GET", f"http://localhost:80{model_type:02d}/v1/models"], check=True, capture_output=True, text=True)
#     print(output.stdout)
#     reasoning_content, real_content = await get_reasoning_and_content(messages, model_type, want_print=True)

# %%
# async way spent time: 3.4s 

# start_time = time.time()
# results = await asyncio.gather(
#     *[get_reasoning_and_content(messages, model_type, want_print=False) for model_type in [5,11]]
# )
# end_time = time.time()
# for model_type, (reasoning_content, real_content) in zip([5,11], results):
#     print(f"=== Model {model_type} ===")
#     print("Reasoning Content:", reasoning_content)
#     print("Real Content:", real_content)
#     print()
# print(f"Total time taken: {end_time - start_time:.2f} seconds")



# %%
async def generate_comparing_table_for_cue_word(cue_word, id):
    
    language = id_to_language_map[id]
    
    if language == 'zh':
        data_dict = data_dict_zh
        system_prompt_1 = "您是一款专为全面探索词语关联而设计的高级语言模型。"
        system_prompt_2 = "你是人工智能助手"

        instruction_prompt_template_1 = "给定一个提示词，你的任务是生成一个与该提示词相关联的全面词汇列表。目标是尽可能涵盖所有相关的语境、用法和含义，避免重复相似的概念。这些词共同提供对所有重要关联的广泛而深刻的表示。专注于揭示与提示词相关的常见和独特的方面，以确保对潜在关联进行平衡而彻底的探索。词语应彼此不同。你的回答只能是相关联的词语列表。不要生成受其他词语存在影响的词语，而是专注于提示词本身\n 提示词：{}"
        instruction_prompt_template_2 = "我们来玩一个词联想实验,给你一个词,你告诉我你立马的联想词有哪些：{}. 请以列表形式返回。"
        
    else:
        data_dict = data_dict_en
        system_prompt_1 = "You are a sophisticated language model designed to explore word associations comprehensively."
        system_prompt_2 = "You are an AI assistant."
        
        instruction_prompt_template_1 = "Given a cue word, your task is to generate a comprehensive list of words associated with the cue word. Aim to cover as many relevant contexts, uses, and meanings as possible without repeating similar concepts. These words together provide a broad and insightful representation of all significant associations. Focus on revealing both common and unique aspects related to the cue word to ensure a balanced and thorough exploration of potential associations. Words should be distinct from each other. Your response shall only be the list of associated words. Do not generate words conditioned on the presence of other words but rather focus on the cue word itself.\n Cue Word: {}"
        
        instruction_prompt_template_2 = "Let's play a word association experiment. Given a cue word, tell me what words you immediately associate with it: {}. Please return them in a list format."
        
   
    # step 1: find the GT associated words
    gt_associated_words = data_dict[cue_word]
    
    # step 2: prompt type 1
    
    messages_type_1 = [
        {"role": "system", "content": system_prompt_1},
        {"role": "user", "content": instruction_prompt_template_1.format(cue_word)}
    ]
    
    # step 2: prompt type 2
    messages_type_2 = [
        {"role": "system", "content": system_prompt_2},
        {"role": "user", "content": instruction_prompt_template_2.format(cue_word)}
    ]
    
    type_1_reasoning_content, type_1_content = await get_reasoning_and_content(messages_type_1, id, want_print=False)
    type_2_reasoning_content, type_2_content = await get_reasoning_and_content(messages_type_2, id, want_print=False)
    
    # step 3: generate dictionary data
    
    data = [
        {
            "Cue Word": cue_word,
            "Ground Truth Associated Words": gt_associated_words,
            "Prompt Type": "Complex",
            "Model Type": model_map[id],
            "Generated Associated Words": type_1_content,
        },
        {
            "Cue Word": cue_word,
            "Ground Truth Associated Words": gt_associated_words,
            "Prompt Type": "Simple",
            "Model Type": model_map[id],
            "Generated Associated Words": type_2_content,
        }
    ]
    return data
    

# %%
# SWOW EN 
SWOW_EN_RESULT_COLLECTION = []
SWOW_ZH_RESULT_COLLECTION = []

# language_to_id_map = {
#     'zh': [1, 3, 5, 7, 9, 11],
#     'en': [2, 4, 6, 8, 10, 12],
#     'us': [2, 4, 6, 8, 10, 12]
# }

language_to_id_map = {
    'zh': [ 5, 11],
}

async def process_en():
    for cue_word in tqdm(data_dict_en.keys()):
        results = await asyncio.gather(
            *[generate_comparing_table_for_cue_word(cue_word, id) for id in language_to_id_map['en']]
        )
        SWOW_EN_RESULT_COLLECTION.extend(results)
        
async def process_us():
    for cue_word in tqdm(data_dict_en.keys()):
        results = await asyncio.gather(
            *[generate_comparing_table_for_cue_word(cue_word, id) for id in language_to_id_map['us']]
        )
        SWOW_EN_RESULT_COLLECTION.extend(results)

async def process_zh():
    for cue_word in tqdm(data_dict_zh.keys()):
        results = await asyncio.gather(
            *[generate_comparing_table_for_cue_word(cue_word, id) for id in language_to_id_map['zh']]
        )
        SWOW_ZH_RESULT_COLLECTION.extend(results)

# await asyncio.gather(process_en(), process_zh())
# await process_zh()

asyncio.run(process_zh())
    


# %%
import pandas as pd

# SWOW_EN_RESULT_COLLECTION_T = []
# for item in SWOW_EN_RESULT_COLLECTION:
#     for subitem in item:
#         SWOW_EN_RESULT_COLLECTION_T.append(subitem)
# SWOW_EN_RESULT_COLLECTION = SWOW_EN_RESULT_COLLECTION_T

SWOW_ZH_RESULT_COLLECTION_T = []
for item in SWOW_ZH_RESULT_COLLECTION:
    for subitem in item:
        SWOW_ZH_RESULT_COLLECTION_T.append(subitem)
        
SWOW_ZH_RESULT_COLLECTION = SWOW_ZH_RESULT_COLLECTION_T


# %%

# Save results to CSV files
# swow_en_df = pd.DataFrame(SWOW_EN_RESULT_COLLECTION, columns=["Cue Word", "Ground Truth Associated Words", "Prompt Type", "Model Type", "Generated Associated Words"])

save_dir = '/data/gpfs/projects/punim2219/LM_with_SWOW/sukaih/cultural-lexis-finetune-llms/notebooks/HLCP-A-eval-12-models/ppo_update_zh_oct_2025'
Path(save_dir).mkdir(parents=True, exist_ok=True)
# swow_en_df.to_csv(os.path.join(save_dir, 'swow_us_results.csv'), index=True)


# # also directly save as pickle file
# swow_en_df.to_pickle(os.path.join(save_dir, 'swow_us_results.pkl'))
# # save as html file
# swow_en_df.to_html(os.path.join(save_dir, 'swow_us_results.html'), index=False)

# with open(os.path.join(save_dir, 'swow_us_results.html'), 'r', encoding='utf-8') as file:
#     html_content = file.read().replace("\\n", "<br>")
# with open(os.path.join(save_dir, 'swow_us_results.html'), 'w', encoding='utf-8') as file:
#     file.write(html_content)

swow_zh_df = pd.DataFrame(SWOW_ZH_RESULT_COLLECTION, columns=["Cue Word", "Ground Truth Associated Words", "Prompt Type", "Model Type", "Generated Associated Words"])
swow_zh_df.to_csv(os.path.join(save_dir, 'swow_zh_results.csv'), index=True)

swow_zh_df.to_pickle(os.path.join(save_dir, 'swow_zh_results.pkl'))

swow_zh_df.to_html(os.path.join(save_dir, 'swow_zh_results.html'), index=False)

with open(os.path.join(save_dir, 'swow_zh_results.html'), 'r', encoding='utf-8') as file:
    html_content = file.read().replace("\\n", "<br>")
with open(os.path.join(save_dir, 'swow_zh_results.html'), 'w', encoding='utf-8') as file:
    file.write(html_content)
    

# %%
SWOW_ZH_RESULT_COLLECTION[0]

# %%



