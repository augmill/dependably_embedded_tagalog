# Information on the project

## Research Question: 
Can we train a language model to understand a low-resource language with only a small dataset of syntactic dependency trees by using synthetic data augmentation and contrastive learning?

## Tasks:

1. Document classification (binary fake news detection on huggingface - jcblaise/fake_news_filipino)

1. Span classification (named entity recognition (NER) on huggingface - ljvmiranda921/tlunified-ner)

1. Text generation (headline generation from news article text on huggingface - LanceBunag/BalitaNLP)

## Models: 

1. MBERT for classification tasks 
(base, MLM-only, CL-only, MLM+CL)

1. Qwen for generation tasks
(CL-finetuned)

1. BASELINE FOR ALL EXPERIMENTS - Qwen family (0.5b, 1.5b, 7b) with prompting

## How to run

In order to reproduce the results from this project, please follow run these files:

1. To reproduce the fine-tuning steps, simply execute the run.sh script in the stage_1 directory, and then the run_cl_training.sh script in the stage_2 directory (scripts require GPU connection). Both scripts call the python scripts in their respective directories, which include the detailed code for dataset loading, model definition, training, evaluation, and logging. The bash scripts require connection to wandb for metrics logging, which can be disabled.

1. For tsne, first install any missing libaries for imports and ensure the approrpiate checked_graphs.jsol is within a folder "data" and contains checked dependency graphs for the task. Run `tsne.py --model {model name}` to obtain the projection for the given step of the archiecture. The options for the model name include "baseline", "stage1", and "CL".

1. For the classification task, run `run.py` in the classifier folder.

## Relevent work

[Link to stage 1 fine-tuned BERT](https://huggingface.co/paulbontempo/bert-tagalog-mlm-stage1) the stage 1 model after unsupervised fine-tuning on fake_new_filipino data without class labels (huggingface)

[Link to stage 2 fine-tuned BERT](https://huggingface.co/paulbontempo/bert-tagalog-dependency-cl) the stage 2 model after contrastive learning from dependency triples on top of the stage 1 embeddings
