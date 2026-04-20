# Ranking Problem Evaluation

> [!WARNING]
> This is conducted on **Test set**

## Qwen 

| label | swow_type | model_class | model_type | Spearman's score   |
|-------|-----------|-------------|------------|--------------------|
| 1     | swow_en   | qwen        | vanilla    | 0.2924648913427331  |
| 2     | swow_en   | qwen        | SFT        | 0.5776125116968458  |
| 3     | swow_en   | qwen        | PPO only   | 0.3218601687565391  |
| 4*    | swow_en   | qwen        | SFT + PPO  | 0.6269135151797346  |
| 5     | swow_zh   | qwen        | vanilla    | 0.2919104921086849  |
| 6     | swow_zh   | qwen        | SFT        | -0.5518954297352932 |
| 7*    | swow_zh   | qwen        | PPO only   | 0.3237818155011084  |
| 8     | swow_zh   | qwen        | SFT + PPO  | -0.3362658578017634 |

## Llama

| label | swow_type | model_class | model_type | Spearman's score   |
|-------|-----------|-------------|------------|--------------------|
| 1     | swow_en   | llama       | vanilla    | 0.24156250719689956 |
| 2     | swow_en   | llama       | SFT        | -0.28324038828036213 |
| 3*    | swow_en   | llama       | PPO only   | 0.2708924450238327  |
| 4     | swow_en   | llama       | SFT + PPO  | -0.32262119685452256 |
| 5     | swow_zh   | llama       | vanilla    | 0.2116635543561007  |
| 6     | swow_zh   | llama       | SFT        | -0.9960205539198455 |
| 7*    | swow_zh   | llama       | PPO only   | 0.2260811639421744  |
| 8     | swow_zh   | llama       | SFT + PPO  | -0.9979645309899143 |

# Word Association Generation Evaluation

## Qwen

| label | model_type     | model_class | swow_type | Prec@5  | Prec@10 | Prec@20 | Prec@30 | Prec@40 | Prec@50 | Spearman |
|-------|----------------|-------------|-----------|---------|---------|---------|---------|---------|---------|----------|
| 1     | vanilla        | qwen        | swow_en   | 0.63398 | 0.50205 | 0.36687 | 0.29031 | 0.23862 | 0.20018 | 0.37266  |
| 2     | SFT            | qwen        | swow_en   | 0.76179 | 0.65171 | 0.4958  | 0.39249 | 0.32799 | 0.28119 | 0.49443  |
| 3     | PPO only       | qwen        | swow_en   | 0.63219 | 0.50415 | 0.36549 | 0.2906  | 0.23822 | 0.19999 | 0.38931  |
| 4*    | SFT + PPO      | qwen        | swow_en   | 0.73577 | 0.65294 | 0.53641 | 0.45165 | 0.38831 | 0.3405  | 0.46247  |
| 5     | vanilla        | qwen        | swow_zh   | 0.48128 | 0.36401 | 0.2547  | 0.19747 | 0.15904 | 0.13237 | 0.28937  |
| 6*    | SFT            | qwen        | swow_zh   | 0.68894 | 0.5593  | 0.40327 | 0.3253  | 0.27972 | 0.25009 | 0.36576  |
| 7     | PPO only       | qwen        | swow_zh   | 0.48507 | 0.36368 | 0.25712 | 0.19634 | 0.15803 | 0.13099 | 0.29121  |
| 8     | SFT + PPO      | qwen        | swow_zh   | 0.59452 | 0.48372 | 0.36448 | 0.3001  | 0.26037 | 0.23396 | 0.32518  |

## Llama

| label | model_type    | model_class | swow_type | Prec@5  | Prec@10 | Prec@20 | Prec@30 | Prec@40 | Prec@50 | Spearman |
|-------|---------------|-------------|-----------|---------|---------|---------|---------|---------|---------|----------|
| 1     | vanilla       | llama       | swow_en   | 0.7548  | 0.60944 | 0.44659 | 0.35541 | 0.29534 | 0.25324 | 0.44843  |
| 2*    | SFT           | llama       | swow_en   | 0.87541 | 0.77366 | 0.62477 | 0.51433 | 0.43703 | 0.3863  | 0.50362  |
| 3     | PPO only      | llama       | swow_en   | 0.75822 | 0.61266 | 0.45086 | 0.35573 | 0.29414 | 0.25245 | 0.45419  |
| 4     | SFT + PPO     | llama       | swow_en   | 0.81611 | 0.70300 | 0.55667 | 0.46052 | 0.39524 | 0.35084 | 0.48155  |
| 5     | vanilla       | llama       | swow_zh   | 0.26029 | 0.18189 | 0.10881 | 0.07594 | 0.05774 | 0.04638 | 0.22356  |
| 6*    | SFT           | llama       | swow_zh   | 0.68973 | 0.55657 | 0.40034 | 0.32006 | 0.27775 | 0.24899 | 0.33558  |
| 7     | PPO only      | llama       | swow_zh   | 0.26168 | 0.18236 | 0.10934 | 0.075809| 0.057507| 0.046236| 0.20959  |
| 8     | SFT + PPO     | llama       | swow_zh   | 0.53894 | 0.43089 | 0.31759 | 0.25545 | 0.22053 | 0.1968  | 0.31472  |

## Observation 

1. The Qwen model with the SFT + PPO combination under Swow_EN outperforms all others in both ranking and generation tasks. Notably, this synergy between SFT and PPO is unique to Qwen Swow_EN and does not appear in other model-type or language combinations.

2. Predicting Swow_ZH is always more difficult than predicting Swow_En

3. Normally. PPO degrades generation performance, SFT degrades ranking performance (except (1))

4. Overall, Qwen surpasses Llama in ranking tasks, demonstrating stronger performance in this area.

5. For generation tasks, Qwen does better than Llama with Swow_ZH, but Llama does better than Qwen with Swow_EN.