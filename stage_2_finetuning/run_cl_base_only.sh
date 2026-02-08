#!/bin/bash
#SBATCH --partition=blanca-blast-lecs
#SBATCH --account=blanca-blast-lecs
#SBATCH --qos=blanca-blast-lecs
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100_2g.20gb:1
#SBATCH --time=04:00:00
#SBATCH --job-name=cl_base_only_training
#SBATCH --output=out-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pabo8622@colorado.edu

echo "========================================"
echo "Starting Contrastive Learning Training (CL-Only Ablation)"
echo "Training base BERT with CL - NO stage 1 MLM finetuning"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Set WandB API key
export WANDB_API_KEY=" "

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
BERT_MODEL="google-bert/bert-base-multilingual-cased"  # Base BERT - NO stage 1 MLM
OUTPUT_DIR="$REPO_DIR/stage_2_finetuning/cl_model_base_only"

# Training hyperparameters - IDENTICAL to MLM+CL run
EPOCHS=50
BATCH_SIZE=24
LEARNING_RATE=3e-5
TEMPERATURE=0.07
PROJECTION_DIM=256
WARMUP_STEPS=300

echo "Training Configuration:"
echo "  Data file: $DATA_FILE"
echo "  BERT model: $BERT_MODEL (base BERT, no stage 1 MLM)"
echo "  Output dir: $OUTPUT_DIR"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Temperature: $TEMPERATURE"
echo "  Projection dim: $PROJECTION_DIM"
echo "  Warmup steps: $WARMUP_STEPS"
echo ""
echo "Note: This is the CL-only ablation (no stage 1 MLM finetuning)"
echo "Note: BERT embeddings will be fine-tuned (not frozen)"
echo ""

# Create output directory
mkdir -p $OUTPUT_DIR

# Change to working directory
cd $REPO_DIR

# Run training
echo "Starting training..."
echo "========================================"

python stage_2_finetuning/train_cl.py \
    --data_file $DATA_FILE \
    --bert_model $BERT_MODEL \
    --output_dir $OUTPUT_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --temperature $TEMPERATURE \
    --projection_dim $PROJECTION_DIM \
    --warmup_steps $WARMUP_STEPS

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
    echo "  Final BERT model: $OUTPUT_DIR/final_model/"
    echo "  This is the CL-only ablation model (base BERT + CL, no stage 1 MLM)"
    echo ""
    echo "To load the model:"
    echo "  from transformers import AutoModel, AutoTokenizer"
    echo "  model = AutoModel.from_pretrained('$OUTPUT_DIR/final_model')"
    echo "  tokenizer = AutoTokenizer.from_pretrained('$OUTPUT_DIR/final_model')"
else
    echo "Training failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
