#!/bin/bash

#SBATCH --job-name ml   ## name that will show up in the queue
#SBATCH --output ./slurmout/ml_slurm-%j.out   ## filename of the output; the %j is equal to jobID; default is slurm-[jobID].out
#SBATCH --ntasks=12  ## number of tasks (analyses) to run
#SBATCH --nodes=4
#SBATCH --cpus-per-task=16  ## the number of threads allocated to each task
#SBATCH --mem-per-cpu=2G   # memory per CPU core
##SBATCH --gpus-per-task=1
##SBATCH --gres=gpu:2  ## number of GPUs per node
#SBATCH --partition=backfill  ## the partitions to run in (comma seperated)
##SBATCH --nodelist=discovery-g11,discovery-g10,discovery-g9,discovery-g8,discovery-g7
#SBATCH --time=0-24:00:00  ## time for analysis (day-hour:min:sec)

## Load modules
## module load cuda/11.2.0-gcc-9.2.0-bdv2ut2
module load cuda/10.1.243-gcc-9.2.0-gl7b4mh
## Insert code, and run your programs here (use 'srun').
module load anaconda3
conda activate py3gpu
time {
for text in {'raw','token'}; do
    for tree in {'none','tree'}; do
        for method in {'SVM','RF'}; do
            for data in {'15_tv','16_tv'}; do
                
                echo tree $tree, method $method, data $data
                srun -N 1 -n 1 -u python main/ml.py --split_type=$data --method=$method --tree=$tree --text=$text &
                srun -N 1 -n 1 -u python main/ml.py --split_type=$data --method=$method --tree=$tree --s --text=$text &
            done
        done
    done
done
echo wait
wait
echo finished
}
