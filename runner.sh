#!/bin/bash

#SBATCH --output=output.txt

#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# # Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

echo "Running on node: $(hostname)"

# Binary or script to execute
python -m src.core.train

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"