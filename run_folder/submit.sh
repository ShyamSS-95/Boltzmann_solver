#!/bin/bash
# Job name:
#SBATCH --job-name=bump_on_tail
#
# Account:
#SBATCH --account=co_astro
#
# Partition:
#SBATCH --partition=savio2_gpu
#
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2 
#SBATCH --ntasks=1
#
# Wall clock limit:
#SBATCH --time=06:00:00
#
## Command(s) to run:
python run_non_linear_solver.py
