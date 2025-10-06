#!/bin/bash 

echo "Downloading WV_Bench data from server 1/2"
rsync -chazvP sukaih@spartan-login1.hpc.unimelb.edu.au:/data/gpfs/projects/punim2219/LM_with_SWOW/kabir/Data/WV_Bench $WORKING_DIR/data/01_raw/

echo "Downloading SWOW data from server 2/2"
rsync -chazvP sukaih@spartan-login1.hpc.unimelb.edu.au:/data/gpfs/projects/punim2219/LM_with_SWOW/chunhua/Data/SWOW $WORKING_DIR/data/01_raw/