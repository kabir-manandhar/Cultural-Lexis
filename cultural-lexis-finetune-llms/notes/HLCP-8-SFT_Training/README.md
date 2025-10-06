> [!IMPORTANT]
> Note that we have the following training requirements:
> 
### Language Specific Training
- **S1 English SWOW (SWOW-EN) with Llama3**: 
  - Train Llama3 using the English version of SWOW (SWOW-EN).
  - Evaluate with English WVS (WVS-EN).

```
***** train metrics *****
  epoch                    =                1.5
  total_flos               =      17159571722GF
  train_loss               =             0.7904
  train_runtime            = 1 day, 15:13:04.62
  train_samples_per_second =              7.383
  train_steps_per_second   =              0.231

***** eval metrics *****
  epoch                   =        1.5
  eval_loss               =     0.4119
  eval_runtime            = 0:00:09.99
  eval_samples_per_second =     80.061
  eval_steps_per_second   =      5.004
```

- **S2 Chinese SWOW (SWOW-ZH) with Qwen**: 
  - Train Qwen using the Chinese version of SWOW (SWOW-ZH).
  - Evaluate with Chinese WVS (WVS-ZH).
```
***** train metrics *****
  epoch                    =                1.5
  total_flos               =      18309192993GF
  train_loss               =             0.8088
  train_runtime            = 1 day, 13:03:30.10
  train_samples_per_second =              7.801
  train_steps_per_second   =              0.244

***** eval metrics *****
  epoch                   =        1.5
  eval_loss               =     0.3237
  eval_runtime            = 0:00:08.45
  eval_samples_per_second =     94.625
  eval_steps_per_second   =      5.914
```


### Cross-lingual Comparison
- **S3 Chinese SWOW (SWOW-ZH) with Llama3**: 
  - Train Llama3 using the Chinese SWOW dataset to assess if it can improve Llama3's understanding of Chinese associations and cultural values.
  - Evaluate with Chinese WVS (WVS-ZH).

```
***** train metrics *****                                                                                                  
  epoch                    =                1.5                                                                            
  total_flos               =      19782856013GF                                                                            
  train_loss               =             0.4932                                                                            
  train_runtime            = 1 day, 16:05:30.87                                                                            
  train_samples_per_second =               7.21                                                                            
  train_steps_per_second   =              0.225   

***** eval metrics *****                                                                                                   
  epoch                   =        1.5                                                                                     
  eval_loss               =     0.2536                                                                                     
  eval_runtime            = 0:00:10.95                                                                                     
  eval_samples_per_second =     73.054                                                                                     
  eval_steps_per_second   =      4.566
```

- **S4 English SWOW (SWOW-EN) with Qwen**: 
  - Train Qwen using the English SWOW dataset to see how it handles English data.
  - Evaluate with English WVS (WVS-EN).
```
***** train metrics *****
  epoch                    =                1.5                                                                            
  total_flos               =      16238636693GF                                                                            
  train_loss               =             3.2146                                                                            
  train_runtime            = 1 day, 12:06:50.79                                                                            
  train_samples_per_second =              8.018                                                                            
  train_steps_per_second   =              0.251
***** eval metrics *****
  epoch                   =        1.5                                                                                     
  eval_loss               =     2.5577                                                                                     
  eval_runtime            = 0:00:07.51                                                                                     
  eval_samples_per_second =    106.463                                                                                     
  eval_steps_per_second   =      6.654     
```