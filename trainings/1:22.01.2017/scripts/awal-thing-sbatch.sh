#!/usr/bin/env bash

#SBATCH --job-name='awal.thing trainer'
#SBATCH -o /Users/akivalipshitz/Developer/netrain/trainings/1:22.01.2017/slurmout/slurm-awal.thing-%j.out
#SBATCH --array=1-1
#SBATCH --time=16:10:00
#SBATCH -p all
#SBATCH -n 4
#SBATCH -N 1

. activate tensorflow
source /Users/akivalipshitz/Developer/netrain/jobsinit.sh
python /Users/akivalipshitz/Developer/netrain/trainings/1:22.01.2017/scripts/awal-thing-initialize.py
