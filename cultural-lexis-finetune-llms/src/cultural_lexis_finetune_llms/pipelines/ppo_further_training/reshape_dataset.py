from random import shuffle
import jsonlines
import os 
from glob import glob
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from datasets import load_dataset
from pathlib import Path

WORKING_DIR = os.environ['WORKING_DIR']

ppo_mcq_params = OmegaConf.load(os.path.join(WORKING_DIR, "conf/base/parameters_ppo_mcq_data_generation.yml"))

def reshape_dataset(ppo_mcq_params):
    """Convert the existing dataset to just having two keys: "input" and "label"

    Args:
        ppo_mcq_params (params): Parameters for the PPO MCQ pipeline, share with Ranking pipeline
    """
    ppo_mcq_params = ppo_mcq_params['ppo_mcq_params']
    print(ppo_mcq_params)
    data_location = ppo_mcq_params["data_location"]
    dataset_name_lst = ppo_mcq_params["dataset_name_lst"]
    split_name = "trl_ranking"
    
    print("data_location", data_location)
    
    for dataset_name in dataset_name_lst:
        print("dataset_name", dataset_name)
        dataset = load_dataset(data_location, dataset_name, split=split_name) 
        # breakpoint()
        
        # gather all the data
        data = []
        for i in tqdm(range(len(dataset))):
            data.append(dataset[i])
            
        # reshape the data
        reshaped_data = []
        for i in tqdm(range(len(data))):
            messages = [
                {"role": "system", "content": data[i]['system']},
                {"role": "user", "content": data[i]['instruction'] + '\n' + data[i]['input']},
            ]
            
            reshaped_data.append({
                "input": messages,
                "label": data[i]['output']
            })
            
        # save the reshaped data
        save_dir = os.path.join(WORKING_DIR, f'data/03_primary/openrlhf_dataset/{dataset_name}')
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # shuffle and split the reshaped data into 0.9 train and 0.1 test
        np.random.seed(42)
        shuffle(reshaped_data)
        train_data = reshaped_data[:int(0.9*len(reshaped_data))]
        test_data = reshaped_data[len(train_data):]
        
        with jsonlines.open(os.path.join(save_dir, f'train.jsonl'), 'w') as writer:
            writer.write_all(train_data)
            
        with jsonlines.open(os.path.join(save_dir, f'test.jsonl'), 'w') as writer:
            writer.write_all(test_data)
            
        print(f"Saved reshaped data for {dataset_name} in {save_dir}")
            
    
    
if __name__ == "__main__":
    reshape_dataset(ppo_mcq_params)