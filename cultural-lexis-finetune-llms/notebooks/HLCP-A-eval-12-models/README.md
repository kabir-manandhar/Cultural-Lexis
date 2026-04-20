## Meeting notes 

Date: 13-06-2025

 
Meeting notes 13th Jun 2025
- Part I: Word association generation:
    - Test set (En, Zh): a list of cue words, with ground-truth associations
    - Generate the associations using 12 models  (En, Zh)   @sukai
    - Compare the differences of the associators from vanilla models and the fine-tuned models:
    - Are they more grounded (less textual book): more Emotion, more concrete, more morally related?
    - Does the fine-tuned model cover more concepts senses (especially the culturally related) ?  (better understanding the concepts)
 
- Part II: Evalaute on World value survey:
    - RAG: put the results @Kabir 
    - Evaluation on the smaller models (put the tables with Earth mover and Jensen–Shannon divergence, and hard accuracy) into the the overleaf first  @Chunhua
    - Evaluation on larger models @Chunhua handle and sync the code with Kabir and handle this to Kabir?
    - /data/projects/punim0478/sukaih/huggingface/hub/models--Qwen--Qwen3-32B
    - Writing

- Side notes:
    - Are the improvements from fine-gunned models significant ? (from Kabir) @chunhua do significance test on the results.  
 
12 models are :
- Qwen vanilla for zh   :8001
- Qwen vanilla for en   :8002
- Qwen SWOW zh          :8003
- Qwen SWOW en          :8004
- Qwen PPO zh           :8005
- Qwen PPO en           :8006

- Llama vanilla for zh  :8007
- Llama vanilla for en  :8008
- Llama SWOW zh         :8009
- Llama SWOW en         :8010
- Llama PPO zh          :8011
- Llama PPO en          :8012

> [!TIP]
> Check `scripts/HLCP-A-eval-12-models/host_flexible_llm.sh` for how to setup the models.
> Check `scripts/HLCP-12-PPO_Training/135_connect_to_169_ports.sh` for how to setup ssh tunnel to the models.

## Results 
The results are separated based on SWOW English and SWOW Chinese. I have saved the pickled DataFrames as `/data/gpfs/projects/punim2219/LM_with_SWOW/sukaih/cultural-lexis-finetune-llms/notebooks/HLCP-A-eval-12-models/swow_{language}_results.pkl`. 

CSV files are also available in the same directory as `swow_{language}_results.csv`.

For a more user-friendly HTML visualisation, I have also saved the results as `swow_{language}_results.html` in the same directory.

## process the results 
Kabir faced some issue with parsing the results, so we have `preprocess_llm_output_into_words.ipynb` to help with the parsing. After getting the `swow_{language}_results.pkl`, you can run the notebook to get the parsed results and get `swow_{language}_results_processed.pkl` which is a dictionary with the following structure:

