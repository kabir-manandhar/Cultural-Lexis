#!/bin/bash

#SBATCH --nodes=1 # change this if you want more than one node

#SBATCH --ntasks=4

#SBATCH --cpus-per-task=2

#SBATCH --time=20-00:00:00

#SBATCH --mem=192G

#SBATCH --partition=deeplearn

#SBATCH -A punim0478

#SBATCH -q gpgpudeeplearn

#SBATCH --gres=gpu:4   # each node has 4 GPUs

#SBATCH --constraint=dlg5|dlg6


# loop forever and sleep 10 sec in each iteration
while true; do
    echo "Sleeping for 1 sec"
    sleep 10
done
