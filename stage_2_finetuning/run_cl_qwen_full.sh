#!/bin/bash
#SBATCH --partition=blanca-blast-lecs
#SBATCH --account=blanca-blast-lecs
#SBATCH --qos=blanca-blast-lecs
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100_2g.20gb:1
#SBATCH --time=08:00:00
#SBATCH --job-name=cl_qwen_full_backprop
#SBATCH --output=out-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pabo8622@colorado.edu

echo "========================================"
echo "Starting CL Training - Qwen-0.5B Full Backprop"
echo "Full LLM forward pass; all parameters trainable"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

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

# Initialize conda
source /curc/sw/anaconda3/latest/etc/profile.d/conda.sh || true
if [ -f "$CONDA_EXE/../etc/profile.d/conda.sh" ]; then
    source "$CONDA_EXE/../etc/profile.d/conda.sh"
fi

# Activate conda environment
conda activate cbert

echo "Environment activated:"
echo "  Python: $(which python)"
echo "  Conda env: $CONDA_DEFAULT_ENV"
echo ""

# Verify GPU
echo "GPU Information:"
nvidia-smi
echo ""

# Verify PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
echo ""

# Repo directory
REPO_DIR="/projects/pabo8622/dependably_embedded_tagalog"

# Configuration
DATA_FILE="$REPO_DIR/data/checked_graphs.jsonl"
LLM_MODEL="Qwen/Qwen2.5-0.5B"
OUTPUT_DIR="$REPO_DIR/stage_2_finetuning/cl_qwen_0.5b_full"

# Training hyperparameters
# Lower lr than embed-only run: full backprop risks overwriting pretrained weights
EPOCHS=50
BATCH_SIZE=16
LEARNING_RATE=1e-5
TEMPERATURE=0.07
PROJECTION_DIM=256
WARMUP_STEPS=500

echo "Training Configuration:"
echo "  Data file: $DATA_FILE"
echo "  LLM model: $LLM_MODEL"
echo "  Output dir: $OUTPUT_DIR"
echo "  Mode: full_backprop (all parameters trainable)"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Temperature: $TEMPERATURE"
echo "  Projection dim: $PROJECTION_DIM"
echo "  Warmup steps: $WARMUP_STEPS"
echo "  Gradient checkpointing: enabled"
echo ""

# Create output directory
mkdir -p $OUTPUT_DIR

# Change to working directory
cd $REPO_DIR

# Run training
echo "Starting training..."
echo "========================================"

python stage_2_finetuning/train_cl_llm.py \
    --data_file $DATA_FILE \
    --llm_model $LLM_MODEL \
    --output_dir $OUTPUT_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --temperature $TEMPERATURE \
    --projection_dim $PROJECTION_DIM \
    --warmup_steps $WARMUP_STEPS \
    --full_backprop

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Job completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "========================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    echo ""
    echo "Model outputs:"
    echo "  Final model: $OUTPUT_DIR/final_model/"
    echo "  Checkpoints every 5 epochs: $OUTPUT_DIR/checkpoint_epoch_*/"
else
    echo "Training failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
