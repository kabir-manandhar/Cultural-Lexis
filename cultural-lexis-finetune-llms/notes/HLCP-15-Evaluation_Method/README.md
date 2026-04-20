## Evaluation that consider both the coverage and the frequency of the associated words 

- **Coverage**: Set the evaluation as `Top-K` evaluation, that is, we only compare with the most frequent `K` words from the ground truth associated words. 
- **Frequency**: We consider giving more score to the words that have higher frequency in the ground truth associated words.

The detailed score design will be in `src/cultural_lexis_finetune_llms/pipelines/finetuning_evaluation/eval_score.py`

