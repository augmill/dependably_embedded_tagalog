#!/bin/bash
#SBATCH --partition=blanca-blast-lecs
#SBATCH --account=blanca-blast-lecs
#SBATCH --qos=blanca-blast-lecs
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100_3g.40gb:1
#SBATCH --time=04:00:00
#SBATCH --job-name=generation_eval
#SBATCH --output=generation_eval_out-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pabo8622@colorado.edu

echo "========================================"
echo "Starting BalitaNLP Generation Task Evaluation"
echo "Perplexity + Headline Generation (Qwen baselines & CL fine-tuned)"
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

# Ensure transformers supports Qwen2 and sentence-transformers is available
pip install -q --upgrade transformers accelerate
pip install -q sentence-transformers sacrebleu rouge-score

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

echo "Evaluation Configuration:"
echo "  Dataset: LanceBunag/BalitaNLP (no-image config, test split)"
echo "  Perplexity samples: 500"
echo "  Headline generation samples: 200"
echo "  Models: Qwen 0.5B/1.5B/7B Instruct baselines + CL embed-only + CL full-backprop"
echo "  Metrics: Perplexity, ChrF++, ROUGE-L, multilingual SBERT cosine similarity"
echo "  Dtype: bfloat16 (all models)"
echo "  GPU: h100_3g.40gb (40GB) — required for Qwen-7B in bfloat16 (~14GB weights + KV cache)"
echo ""

# Change to repo directory so relative imports resolve correctly
cd $REPO_DIR

# Run evaluation
echo "Starting evaluation..."
echo "========================================"

python generation_task/run_generation.py

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Job completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "========================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Evaluation completed successfully!"
else
    echo "Evaluation failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
