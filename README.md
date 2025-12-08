# Information on the project

## How to run

In order to reproduce this project from

To reproduce the fine-tuning steps, simply execute the run.sh script in the stage_1 directory, and then the run_cl_training.sh script in the stage_2 directory (scripts require GPU connection). Both scripts call the python scripts in their respective directories, which include the detailed code for dataset loading, model definition, training, evaluation, and logging. The bash scripts require connection to wandb for metrics logging, which can be disabled.

## Relevent work

Link to finetuned BERT model trained on fake_new_filipino data (huggingface): https://huggingface.co/paulbontempo/bert-tagalog-mlm-stage1