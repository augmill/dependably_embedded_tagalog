#!/bin/bash

# Upload Stage 2 Contrastive Learning model to Hugging Face Hub
# This version uploads the BERT model with fine-tuned embeddings from CL training

set -e  # Exit on error

echo "========================================"
echo "Uploading CL-Finetuned BERT to HF Hub"
echo "========================================"

# Configuration - UPDATE THESE
HF_USERNAME="paulbontempo"
MODEL_NAME="bert-tagalog-dependency-cl"
MODEL_PATH="/projects/pabo8622/NeSy/CL/cl_model_v3/final_model"
STAGE1_MODEL="paulbontempo/bert-tagalog-mlm-stage1"
MAKE_PRIVATE=false

echo ""
echo "Configuration:"
echo "  Username: $HF_USERNAME"
echo "  Model name: $MODEL_NAME"
echo "  Model path: $MODEL_PATH"
echo "  Stage 1 model: $STAGE1_MODEL"
echo "  Private: $MAKE_PRIVATE"
echo ""

# Load conda environment (skip if already active)
if [[ "$CONDA_DEFAULT_ENV" != "cbert" ]]; then
    echo "Loading environment..."
    module load anaconda
    conda activate cbert
else
    echo "Environment already active: $CONDA_DEFAULT_ENV"
fi

# Install huggingface_hub if not already installed
echo "Installing/updating huggingface_hub..."
pip install -q huggingface_hub

# Check if already logged in
TOKEN_FOUND=false
if [ -f ~/.huggingface/token ]; then
    TOKEN_FOUND=true
    echo "✅ Token found in ~/.huggingface/token"
elif [ -f ~/.cache/huggingface/token ]; then
    TOKEN_FOUND=true
    echo "✅ Token found in ~/.cache/huggingface/token"
elif [ -f /projects/$USER/.cache/pip/huggingface/token ]; then
    TOKEN_FOUND=true
    echo "✅ Token found in /projects/$USER/.cache/pip/huggingface/token"
elif [ -f /projects/$USER/.cache/huggingface/token ]; then
    TOKEN_FOUND=true
    echo "✅ Token found in /projects/$USER/.cache/huggingface/token"
fi

if [ "$TOKEN_FOUND" = false ]; then
    echo ""
    echo "⚠️  You need to login to Hugging Face first!"
    echo ""
    echo "Please run this command and follow the prompts:"
    echo "  huggingface-cli login"
    echo ""
    echo "Get your token from: https://huggingface.co/settings/tokens"
    echo ""
    exit 1
fi

# Check if model directory exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Error: Model directory not found at $MODEL_PATH"
    exit 1
fi

echo "✅ Model directory found"

# Check for required BERT model files
echo "Checking for required files..."
REQUIRED_FILES=("config.json" "pytorch_model.bin" "tokenizer_config.json" "vocab.txt")
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$MODEL_PATH/$file" ]; then
        echo "❌ Error: $file not found in $MODEL_PATH"
        exit 1
    fi
done
echo "✅ Required BERT files found"

# Export environment variables for Python script
export HF_USERNAME
export MODEL_NAME
export MODEL_PATH
export STAGE1_MODEL
export MAKE_PRIVATE

# Create the Python upload script
echo ""
echo "Creating upload script..."

cat > /tmp/upload_cl_bert_to_hf.py << 'EOF'
from huggingface_hub import HfApi
import sys
import os

username = os.environ['HF_USERNAME']
model_name = os.environ['MODEL_NAME']
model_path = os.environ['MODEL_PATH']
stage1_model = os.environ['STAGE1_MODEL']
make_private = os.environ['MAKE_PRIVATE'].lower() == 'true'

print(f"\n{'='*50}")
print(f"Uploading to: {username}/{model_name}")
print(f"Private: {make_private}")
print(f"{'='*50}\n")

api = HfApi()

print("Creating repository on Hugging Face Hub...")
try:
    repo_url = api.create_repo(
        repo_id=f"{username}/{model_name}",
        repo_type="model",
        private=make_private,
        exist_ok=True,
    )
    print(f"✅ Repository ready: {repo_url}")
except Exception as e:
    print(f"Repository creation: {e}")

print("\nCreating model card...")
readme_content = f"""---
license: mit
language:
- tl
tags:
- tagalog
- dependency-parsing
- contrastive-learning
- bert
- syntax
- low-resource
base_model: {stage1_model}
library_name: transformers
---

# Tagalog BERT with Dependency-Aware Contrastive Learning

This is a BERT model for Tagalog with token embeddings fine-tuned using contrastive learning on dependency parse tree structures.

## Model Description

- **Base Model:** [{stage1_model}](https://huggingface.co/{stage1_model}) (we fine-tuned the stage_1 model itself from base BERT on the FakeNewsFilipino dataset)
- **Language:** Tagalog (Filipino)
- **Training Approach:** Two-stage fine-tuning for low-resource language processing
  1. **Stage 1:** Masked Language Modeling (MLM) on Tagalog corpus (FakeNewsFilipino)
  2. **Stage 2:** Contrastive learning with InfoNCE loss on dependency parse triples corpus (UD-Ugnayan)

## Our Contributions

We use a novel approach to encode syntactic structure directly into token embeddings:
- Dependency triples (head, relation, dependent) were extracted from 94 UD-annotated Tagalog sentences
- Contrastive learning with InfoNCE loss trained tokens to cluster by their syntactic roles
- Tokens appearing as heads of the same dependency relation become similar in embedding space
- This improves downstream NLP task performance for low-resource Tagalog

## Architecture

Standard BERT architecture with fine-tuned token embeddings:
- **Hidden size:** 768
- **Attention heads:** 12
- **Layers:** 12
- **Vocabulary size:** ~50,000 tokens (WordPiece)

The contrastive learning stage used:
- **Loss:** InfoNCE (temperature=0.07)
- **Projection dimension:** 256
- **Training epochs:** 50
- **Final loss:** 0.076

## Usage

This is a standard HuggingFace BERT model and can be used like any other BERT:

```python
from transformers import AutoModel, AutoTokenizer

# Load model and tokenizer
model = AutoModel.from_pretrained("{username}/{model_name}")
tokenizer = AutoTokenizer.from_pretrained("{username}/{model_name}")

# Use for embeddings
text = "Magandang umaga sa lahat"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

# Get token embeddings
token_embeddings = outputs.last_hidden_state
```

### For Downstream Tasks

Fine-tune on your Tagalog NLP task:

```python
from transformers import AutoModelForSequenceClassification

# For classification tasks
model = AutoModelForSequenceClassification.from_pretrained(
    "{username}/{model_name}",
    num_labels=3
)

# Train on your task
# ...
```

## Training Details

### Stage 2: Contrastive Learning
- **Dataset:** 94 Tagalog sentences with dependency annotations
- **Positive samples:** ~600 true dependency triples
- **Negative samples:** ~10,000 artificially generated incorrect triples
- **Batch strategy:** Relation-aware batching for efficient positive pair sampling
- **Optimizer:** AdamW (lr=3e-5, weight_decay=0.01)
- **Warmup steps:** 300
- **Training time:** ~30 minutes on H100 GPU

### Contrastive Learning Strategy
- **Positive pairs:** Triples with the same dependency relation from true parses
- **Negative pairs:** Artificially created grammatically incorrect triples OR triples with different relations
- **Goal:** Cluster tokens by syntactic role to improve representation quality

## Evaluation

This model is designed as a pre-trained base for downstream Tagalog NLP tasks. The quality of embeddings can be evaluated through:
- Dependency parsing accuracy
- Named entity recognition
- Sentiment analysis
- Other token-level classification tasks

## Limitations

- Trained on only 94 sentences with dependency annotations (very small dataset)
- May not generalize to all Tagalog language varieties
- Best used as a starting point for further task-specific fine-tuning

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{{bert-tagalog-dependency-cl,
  author = {{Paul Bontempo}},
  title = {{Tagalog BERT with Dependency-Aware Contrastive Learning}},
  year = {{2025}},
  publisher = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/{username}/{model_name}}}}}
}}
```

## Acknowledgments

- Built on top of Stage 1 MLM training: [{stage1_model}](https://huggingface.co/{stage1_model})
- Developed at University of Colorado Boulder
- Part of neural-symbolic (NeSy) research for low-resource language processing
"""

readme_path = os.path.join(model_path, "README.md")
with open(readme_path, 'w', encoding='utf-8') as f:
    f.write(readme_content)
print("✅ Model card created")

# Find all files to upload
import glob
print("\nFinding files to upload...")
all_files = glob.glob(os.path.join(model_path, "*"))
files_to_upload = [os.path.basename(f) for f in all_files if os.path.isfile(f)]

print(f"Found {len(files_to_upload)} files:")
for f in files_to_upload:
    print(f"  - {f}")

print("\nUploading files...")
for filename in files_to_upload:
    file_path = os.path.join(model_path, filename)
    
    if not os.path.exists(file_path):
        print(f"⚠️  Warning: {filename} not found, skipping")
        continue
    
    print(f"  Uploading {filename}...")
    try:
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=filename,
            repo_id=f"{username}/{model_name}",
            repo_type="model",
            commit_message=f"Upload {filename}"
        )
        print(f"  ✅ {filename} uploaded")
    except Exception as e:
        print(f"  ❌ Error uploading {filename}: {e}")
        sys.exit(1)

print("\n" + "="*50)
print("✅ Upload complete!")
print("="*50)
print(f"\nModel available at: https://huggingface.co/{username}/{model_name}")
print(f"\nTo use:")
print(f"  from transformers import AutoModel, AutoTokenizer")
print(f"  model = AutoModel.from_pretrained('{username}/{model_name}')")
print(f"  tokenizer = AutoTokenizer.from_pretrained('{username}/{model_name}')")
EOF

# Run the upload
echo "Starting upload..."
python /tmp/upload_cl_bert_to_hf.py

# Cleanup
rm /tmp/upload_cl_bert_to_hf.py

echo ""
echo "========================================"
echo "Upload process complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Visit https://huggingface.co/$HF_USERNAME/$MODEL_NAME"
echo "2. Verify the model card and files"
echo "3. Test loading: AutoModel.from_pretrained('$HF_USERNAME/$MODEL_NAME')"
echo ""
