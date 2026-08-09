#!/bin/bash
#SBATCH -A MED117
#SBATCH -J preprocess_exposome
#SBATCH -N 1
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -q debug

source /lustre/orion/med117/proj-shared/exposomedata/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/med117/proj-shared/exposomedata/miniconda3/envs/pm

python ../matey/data_utils/preprocessexposome.py --timescale static --overwrite 
python ../matey/data_utils/preprocessexposome.py --timescale daily --overwrite


