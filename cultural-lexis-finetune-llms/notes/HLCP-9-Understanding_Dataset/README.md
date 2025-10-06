## Overview of the Experiment

1. We want to fine-tune the LLMs to predict the SWOW dataset when they are provided with a target word. 
2. Once we have fine-tuned the model, we believe that the model will be able to carry the cultural bias that is embedded in the SWOW dataset.
    - We assume that SWOW English dataset will carry the cultural bias of the western world (US, UK, etc.) and SWOW Chinese dataset will carry the cultural bias of the Asian world (China, Japan, etc.)
3. We will then use the fine-tuned model to answer the World Value Benchmark ([WV_Bench](https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp)) and see if the model can behave in a way that is consistent with the cultural bias of the participants who come from different countries.


> [!WARNING]
> It seems crucial to design the prompts for fine-tuning the model. A very straightforward prompt should look like this:


```plaintext
## Context:
(Explain what is forward and backward word association and related words (It looks like Mandarin does not provide related words for the target word))
...
(Provide the target word)
The target word is "apple".

## Question:
What is the forward word you can think of when you hear "apple"? 

## Answer:
[The answer should be from the SWOW dataset]

## Question:
What is the backward word you can think of when you hear "apple"?

## Answer:
[The answer should be from the SWOW dataset]
```

> [!TIP]
> Besides only asking for the forward and backward word, we can also ask for predicting the verbalised score of the association. It is known that LLMs are also capable of predicting the scores [1], we can see if predicting the score can help the model to carry the cultural bias more effectively.

## TODO 
- [x] Ask Huahua to provide access to the SWOW dataset in Spartan server.
- [x] Wait for the Mandarin SWOW dataset to be ready.

## Reference 

[1]. Hou, Yupeng, et al. "Large language models are zero-shot rankers for recommender systems." European Conference on Information Retrieval. Cham: Springer Nature Switzerland, 2024.