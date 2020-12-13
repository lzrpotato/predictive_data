#!/bin/bash

#SBATCH --job-name gpu   ## name that will show up in the queue
#SBATCH --output ./slurmout/slurm-%j.out   ## filename of the output; the %j is equal to jobID; default is slurm-[jobID].out
#SBATCH --ntasks=1  ## number of tasks (analyses) to run
#SBATCH --cpus-per-task=56  ## the number of threads allocated to each task
#SBATCH --mem-per-cpu=2G   # memory per CPU core
#SBATCH --gres=gpu:1  ## number of GPUs per node
#SBATCH --partition=epscor  ## the partitions to run in (comma seperated)
#SBATCH --time=0-24:00:00  ## time for analysis (day-hour:min:sec)

## Load modules
module load cuda/10.1.243-gcc-9.2.0-gl7b4mh
## Insert code, and run your programs here (use 'srun').
source venv/bin/activate
srun -u python train.py