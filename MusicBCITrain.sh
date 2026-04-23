#!/bin/bash
#SBATCH --job-name=MusicBCI_train
#SBATCH --output=sbatch_outputs/MusicBCI_%j.out
#SBATCH --error=sbatch_outputs/MusicBCI_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

module purge
module load conda

# Initialize conda
source /apps/conda/miniforge3/24.11.3/etc/profile.d/conda.sh

conda activate musicbci

# Run training script
python -u training_pipe.py

echo "===== JOB END ====="

conda deactivate
