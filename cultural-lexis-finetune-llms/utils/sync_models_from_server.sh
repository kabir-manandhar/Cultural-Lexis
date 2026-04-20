#!/bin/bash

rsync -chavP sukaih@spartan-login1.hpc.unimelb.edu.au:/data/projects/punim0478/sukaih/Sukai_Project/huahua/data/07_model_output /home/sukaih/Extrastorage/HuaHuaProject/cultural-lexis-finetune-llms/data/


# here is also a script that should run at server to directly cp the trained model to the place to want 

# rsync -chavP /data/projects/punim0478/sukaih/Sukai_Project/huahua/data/07_model_output /data/gpfs/projects/punim2219/LM_with_SWOW/sukaih/cultural-lexis-finetune-llms/data/