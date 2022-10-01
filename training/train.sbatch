#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --job-name=pifno
#SBATCH --array=1-10
. ~/.bashrc
conda activate torch171_cuda102
cd /scratch/btt1/resinv/pi-fno/expt-4/
python -u train_pifno-forward.py $SLURM_ARRAY_TASK_ID > train_pifno-forward-$SLURM_ARRAY_TASK_ID.out
