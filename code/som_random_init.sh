#!/bin/tcsh

#

#SBATCH --partition=fdr  

#SBATCH -n 1

#SBATCH --mem-per-cpu=10000

#SBATCH --time=20:00:00       # format is DAYS-HOURS:MINUTES:SECONDS

#SBATCH --job-name=job

#SBATCH --output=outputs/som_random_init.out

module load python/3.7.4
python som_random_init.py