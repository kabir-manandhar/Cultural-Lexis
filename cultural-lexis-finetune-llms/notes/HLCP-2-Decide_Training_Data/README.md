## Decide the training data and the model combination

### Language Specific Training
- **English SWOW (SWOW-EN) with Llama3**: 
  - Train Llama3 using the English version of SWOW (SWOW-EN).
  - Evaluate with English WVS (WVS-EN).
  
- **Chinese SWOW (SWOW-ZH) with Qwen**: 
  - Train Qwen using the Chinese version of SWOW (SWOW-ZH).
  - Evaluate with Chinese WVS (WVS-ZH).

### Cross-lingual Comparison
- **Chinese SWOW (SWOW-ZH) with Llama3**: 
  - Train Llama3 using the Chinese SWOW dataset to assess if it can improve Llama3's understanding of Chinese associations and cultural values.
  - Evaluate with Chinese WVS (WVS-ZH).

- **English SWOW (SWOW-EN) with Qwen**: 
  - Train Qwen using the English SWOW dataset to see how it handles English data.
  - Evaluate with English WVS (WVS-EN).

### Considerations
- **Translation Impact**: 
  - The translation of SWOW datasets between languages might lead to loss of cultural connotations. It's worth considering the cost and accuracy of translation. Therefore it is **not** recommend to **translate** the SWOW dataset. 

- **Model Suitability**: 
  - Research into which models are most effective for processing Chinese language data is necessary. See what models are better than Qwen for Chinese language processing.

