#!/bin/bash
#SBATCH --partition=blanca-blast-lecs
#SBATCH --account=blanca-blast-lecs
#SBATCH --qos=blanca-blast-lecs
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100_2g.20gb:1
#SBATCH --time=04:00:00
#SBATCH --job-name=bert_tagalog_mlm
#SBATCH --output=out-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pabo8622@colorado.edu

# Disable WandB logging
export WANDB_MODE=disabled

# Set temp directories
export TMPDIR=/projects/$USER/tmp
export TEMP=/projects/$USER/tmp
export TMP=/projects/$USER/tmp

# Clean Python environment variables
unset PYTHONPATH
unset PIP_PREFIX
unset PYTHONNOUSERSITE

# Load modules
module purge
module load anaconda

# Init conda for bash script
source /curc/sw/anaconda3/latest/etc/profile.d/conda.sh

# Activate conda environment
conda activate cbert

# DEBUG: Check if environment activated
echo "Active conda environment: $CONDA_DEFAULT_ENV"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"
conda list | grep torch

# Repo directory
REPO_DIR=/projects/pabo8622/dependably_embedded_tagalog

# Change to the stage 1 finetuning directory
cd $REPO_DIR/stage_1_finetuning

# Verify GPU and PyTorch
nvidia-smi
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# Run script
python bert_genfinetune.py
