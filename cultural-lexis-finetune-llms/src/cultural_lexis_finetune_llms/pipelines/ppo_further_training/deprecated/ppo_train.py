"""
This is a boilerplate pipeline 'ppo_further_training'
generated using Kedro 0.19.9
"""

from pathlib import Path
import shutil
import pickle
from accelerate import PartialState
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, set_seed, DummyOptim, DummyScheduler
from datasets import load_dataset, load_from_disk
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    DataCollatorWithPadding,
)
import os 
from trl import ModelConfig, RLOOConfig, ScriptArguments
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from omegaconf import OmegaConf
from trl.trainer.utils import prepare_deepspeed, exact_div
from torch.utils.data import DataLoader

from src.cultural_lexis_finetune_llms.pipelines.ppo_further_training.ppo_trainer_c import RLOOTrainer

from src.cultural_lexis_finetune_llms.pipelines.ppo_further_training.swow_manual_reward import swow_manual_reward_frequency


from src.cultural_lexis_finetune_llms.pipelines.ppo_further_training.dataset_setup import lf_ver_prepare_dataset, TRAIN_CACHE_DIR, EVAL_CACHE_DIR

from functools import partial

"""
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    examples/scripts/rloo/rloo_tldr.py \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
    --dataset_test_split validation \
    --output_dir models/minimal/rloo_tldr \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --total_episodes 1000000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --local_rollout_forward_batch_size 16 \
    --missing_eos_penalty 1.0 \
    --stop_token eos
"""


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, RLOOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    # remove output_dir if exists
    shutil.rmtree(training_args.output_dir, ignore_errors=True)
    accelerator = PartialState()
    ds_config_fp = os.path.join(os.environ['WORKING_DIR'], 'scripts/HLCP-11-PPO_Training/configs/deepspeed_zero2.yaml')
    ds_config = OmegaConf.load(ds_config_fp)
    training_args.world_size = ds_config['num_processes']
    
    template = os.environ['LF_TEMPLATE'] # 'llama3' or 'qwen'
    subset = os.environ['LF_SUBSET'] # 'swow_en' or 'swow_zh' 
    split = os.environ['LF_SPLIT'] # 'trl'
    lf_dataset_dir = os.path.join(os.environ['WORKING_DIR'], os.environ['LF_DATASET_DIR'])
    training_args.exp_name = f'ppo_further_training_{subset}_{split}_{template}'
    
    custom_config = OmegaConf.load(os.path.join(os.environ['WORKING_DIR'], 'conf/base/parameters_ppo_further_training.yml'))
    custom_config = OmegaConf.to_container(custom_config, resolve=True)
    accelerator.print(f"=== Custom Config: {custom_config}")
    save_steps = custom_config['save_steps']
    training_args.save_steps = save_steps
    save_total_limit = custom_config['save_total_limit']
    training_args.save_total_limit = save_total_limit
    training_args.token_level_kl = custom_config['token_level_kl']
    training_args.normalize_advantage = custom_config['normalize_advantage']
    training_args.cliprange = custom_config['cliprange']
    training_args.normalize_reward = custom_config['normalize_reward']
    training_args.reward_clip_range = custom_config['reward_clip_range']
    training_args.missing_eos_penalty = custom_config['missing_eos_penalty']
    
    ################
    # Model & Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, padding_side="left", trust_remote_code=model_args.trust_remote_code
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
        
    # reward_model = AutoModelForSequenceClassification.from_pretrained(
    #     training_args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
    # )
    
    ################
    # Reward Function
    ################
    
    # load pickled associated_word_freq_dict
    dict_fp = os.path.join(os.environ['WORKING_DIR'], f'data/02_intermediate/nested_{subset}.pkl')
    with open(dict_fp, 'rb') as f:
        associated_word_freq_dict = pickle.load(f)
        
    reward_function = partial(swow_manual_reward_frequency,
                                associated_word_freq_dict=associated_word_freq_dict)
    
    ################
    # Dataset
    ################
    accelerator.print("=== Preparing Dataset")
    # *-- OLD CODE --
    # with accelerator.local_main_process_first():
    #     train_dataset, eval_dataset = lf_ver_prepare_dataset(
    #         subset=subset,
    #         split=split,
    #         template=template,
    #         lf_dataset_dir=lf_dataset_dir,
    #         model_args=model_args,
    #         training_args=training_args,
    #         tokenizer=tokenizer,
    #     )
    # *-- END OF OLD CODE --
    
    # *-- NEW CODE --
    train_dataset = load_dataset(os.path.join(os.environ['WORKING_DIR'], 'data/03_primary/llm_swow_finetune_dataset'), subset, split=split)
    eval_dataset = load_dataset(os.path.join(os.environ['WORKING_DIR'], 'data/03_primary/llm_swow_finetune_dataset'), subset, split='test')

    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""

        def tokenize(element):
            messages = [
                {"role": "system", "content": element['system']},
                {"role": "user", "content": element['instruction'] + '\n' + element['input']},
            ]
            input_ids = tokenizer.apply_chat_template(
                messages,
                padding=False,
                add_generation_prompt=True,
            )
            return {"input_ids": input_ids, "lengths": len(input_ids)}

        return dataset.map(
            tokenize,
            remove_columns=dataset.column_names,
            num_proc=training_args.dataset_num_proc,
        )

    # Compute that only on the main process for faster data processing.
    # see: https://github.com/huggingface/trl/pull/1255
    with accelerator.local_main_process_first():
        train_dataset = prepare_dataset(train_dataset, tokenizer)
        eval_dataset = prepare_dataset(eval_dataset, tokenizer)

    # print some train_dataset
    if accelerator.is_local_main_process:
        tested_text = tokenizer.batch_decode(train_dataset[:2]["input_ids"])
        accelerator.print(f"=== Example of the first training example: {tested_text}")

    # *-- END OF NEW CODE --
    assert train_dataset[0]["input_ids"][-1] != tokenizer.eos_token_id, "The last token should not be an EOS token"
    
    ################
    # Model
    ################
    accelerator.print("=== Setting up Policy Model")
    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code,
        low_cpu_mem_usage=True
    )
    
    # !-- ADDITIONAL CODE --
    accelerator = Accelerator(gradient_accumulation_steps=training_args.gradient_accumulation_steps)
    
    
    training_args.local_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * training_args.num_mini_batches
    )
    
    local_dataloader_batch_size = exact_div(
        training_args.local_batch_size, training_args.rloo_k, "`local_batch_size` must be a multiple of rloo_k"
    )  # RLOO logic: needed because RLOO repeats the same prompt args.rloo_k times
    data_collator = DataCollatorWithPadding(tokenizer)
    dataloader = DataLoader(
        train_dataset,
        batch_size=local_dataloader_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        drop_last=True,  # needed; otherwise the last batch will be of ragged shape
    )
    torch.manual_seed(training_args.seed)
    accelerator.print("=== Preparing Models with Accelerator ===")
    optimizer = torch.optim.AdamW(policy.parameters(), lr=training_args.learning_rate)
    policy, optimizer, dataloader = accelerator.prepare(policy, optimizer, dataloader)
    
    accelerator.print("=== Setting up Ref Policy Model")
    ref_policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code,
        low_cpu_mem_usage=True
    )
    
    # !-- ADDITIONAL CODE --
    is_deepspeed_enabled = getattr(accelerator.state, "deepspeed_plugin", None) is not None
    
    if is_deepspeed_enabled:
    
        accelerator.print("=== Preparing DeepSpeed for Ref Policy ===")
        ref_policy = prepare_deepspeed(
            ref_policy, training_args.per_device_train_batch_size, training_args.fp16, training_args.bf16
        ) # 
    # !-- END OF ADDITIONAL CODE --
    
    ################
    # Training
    ################
    trainer = RLOOTrainer(
        config=training_args,
        processing_class=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_function=reward_function,
        template_str = template,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        accelerator=accelerator,
        train_dataloader=dataloader,
        processed_optimizer=optimizer,
    )
    accelerator.print("=== Training")
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    trainer.generate_completions()