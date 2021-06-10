#!/bin/bash
# launch jupyter

#SBATCH -J jupyter
#SBATCH --time=7:00:00
#SBATCH --mem=6G
#SBATCH -o jupyter.out

# Setup Environment
module load anaconda3/2020.11
source activate py3.8_som

jupyter lab --no-browser --ip "*" \
            --notebook-dir /home/shray/project_2