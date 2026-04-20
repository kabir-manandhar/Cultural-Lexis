MCQ_DATASET_TEMPLATE_EN = {
    "system": "You are a sophisticated language model designed to explore word associations comprehensively.",
    "instruction": "Given the cue word, select the option ({option_choice}) that contains the words most strongly and directly associated with it. Choose the option that best represents the most common and immediate associations with the cue word.\n\n**Think step by step to analyze each option and determine the best choice.**\n\n**Finally, output your answer in the following format:** `Final Answer: [Option Letter]` **(for example, `Final Answer: [A]`)**.  **Ensure that your the end of response contains only the final answer in the specified format, without any additional text.**",
    "input": "{cue_word}",
    "output": "{answer}", # Expected correct output - you would use this for evaluation/reward calculation
    "choices": [
        {
            "letter": "{option_letter}",
            "text": "{answer_choice}",
        },
        
    ]
}

RANKING_DATASET_TEMPLATE_EN = {
    "system": "You are a sophisticated language model designed to explore word associations and rank them based on the strength of their relationship with a cue word.",
    "instruction": """Given the cue word, rank the following associated words from the most strongly related (rank 1) to the least strongly related (rank K, where K is the number of associated words).

Think step by step, comparing each associated word to the others to determine their relative strength of association with the cue word.

**Your final answer should at the end of the response and be in the following format:**

```
Final Ranking:
Rank 1: [Associated Word]
Rank 2: [Associated Word]
...
Rank K: [Associated Word]
```
""",
    "input": "{cue_word_with_associated_words_data}",
    "output": "{answer}", # Expected correct output - you would use this for evaluation/reward calculation
}

RANKING_DATASET_TEMPLATE_ZH ={
    "system": "你是一个设计精良的语言模型，旨在全面探索词语关联性并根据与提示词的关联强度对其进行排名。",
    "instruction": """给定一个提示词，请将以下关联词从最强关联（排名1）到最弱关联（排名K，其中K是关联词的数量）进行排名。
    
逐步比较每个关联词与其他关联词，以确定它们与提示词的相对关联强度。

**你的最终答案应该在回答的最后，并且应该采用以下格式：**

```
最终排名:
排名1: [关联词]
排名2: [关联词]
...
排名K: [关联词]
```
""",
    "input": "{cue_word_with_associated_words_data}",
    "output": "{answer}", # 预期正确输出 - 您可以将其用于评估/奖励计算
}

MCQ_DATASET_TEMPLATE_ZH = {
    "system": "你是一个设计精良的语言模型，旨在全面探索词语关联性。",
    "instruction": "给定一个提示词，请选择包含与该词最直接且关联最强的词语的选项（{option_choice}）。选择最能代表该提示词最常见和直接关联的选项。\n\n**请逐步分析每个选项，以确定最佳选择。**\n\n**最后，请按照以下格式输出答案：** `最终答案: [选项字母]` **（例如，`最终答案: [A]`）**。**确保你的回答结尾只包含指定格式的最终答案，不包含任何额外文本。**",
    "input": "{cue_word}",
    "output": "{answer}",  # 预期正确输出 - 您可以将其用于评估/奖励计算
    "choices": [
        {
            "letter": "{option_letter}",
            "text": "{answer_choice}",
        },
    ]
}