#!/bin/bash

rsync -chavP --exclude-from=rsync_exclude_files.txt /home/sukaih/Extrastorage/HuaHuaProject/cultural-lexis-finetune-llms sukaih@spartan-login1.hpc.unimelb.edu.au:/data/gpfs/projects/punim2219/LM_with_SWOW/sukaih/



# rsync -chavP /home/sukaih/Extrastorage/HuaHuaProject/cultural-lexis-finetune-llms/data/03_primary/openrlhf_dataset sukaih@spartan-login1.hpc.unimelb.edu.au:/data/gpfs/projects/punim2219/LM_with_SWOW/sukaih/cultural-lexis-finetune-llms/data/03_primary/