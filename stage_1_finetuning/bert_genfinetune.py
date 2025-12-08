"""
BERT Fine-tuning on Tagalog Text using MLM
Stage 1: General language adaptation before contrastive learning with parse trees
"""

import os
import torch
import wandb
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    set_seed
)
from typing import Dict, Any
import numpy as np

# Set seed for reproducibility
set_seed(42)

class BERTMLMTrainer:
    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        dataset_name: str = "jcblaise/fake_news_filipino",
        output_dir: str = "./bert_tagalog_mlm",
        max_length: int = 512,
        train_split: float = 0.8,
        val_split: float = 0.1,
        # test_split will be 0.1 (remainder)
    ):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.max_length = max_length
        self.train_split = train_split
        self.val_split = val_split
        
        # Initialize wandb
        wandb.init(
            project="bert-tagalog-mlm",
            name="stage1-general-finetuning",
            config={
                "model": model_name,
                "dataset": dataset_name,
                "max_length": max_length,
                "train_split": train_split,
                "val_split": val_split,
            }
        )
        
    def load_and_prepare_data(self):
        """Load dataset and create train/val/test splits"""
        print("Loading dataset...")
        
        # Try multiple locations for the dataset
        possible_paths = [
            "fnf_data.csv",  # Your uploaded CSV file
            "fake_news_filipino_data/dataset",  # Processed HF dataset
            "fake_news_filipino_data/fnf_data.csv",  # Alternative location
        ]
        
        full_dataset = None
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Found dataset at: {path}")
                
                # Load based on file type
                if path.endswith('.csv'):
                    print(f"Loading CSV file: {path}")
                    dataset = load_dataset('csv', data_files=path)
                    full_dataset = dataset['train']
                    print(f"Loaded {len(full_dataset)} samples from CSV")
                    break
                else:
                    # It's a processed HuggingFace dataset
                    print(f"Loading from processed dataset: {path}")
                    dataset = load_from_disk(path)
                    
                    # If dataset has train/valid/test splits, combine them
                    if isinstance(dataset, dict) or hasattr(dataset, 'keys'):
                        from datasets import concatenate_datasets
                        all_data = []
                        for split_name in dataset.keys():
                            all_data.append(dataset[split_name])
                        full_dataset = concatenate_datasets(all_data)
                        print(f"Combined {len(dataset.keys())} splits into {len(full_dataset)} samples")
                    else:
                        full_dataset = dataset
                    break
        
        if full_dataset is None:
            print(f"\n❌ Dataset not found in any of these locations:")
            for path in possible_paths:
                print(f"  - {path}")
            print("\nPlease ensure fnf_data.csv is in the same directory as this script.")
            raise RuntimeError("Dataset not found. Please check file location.")
        
        print(f"Dataset loaded successfully with {len(full_dataset)} samples")
        
        # Calculate split sizes
        total_size = len(full_dataset)
        train_size = int(self.train_split * total_size)
        val_size = int(self.val_split * total_size)
        
        # Create splits
        train_test_split = full_dataset.train_test_split(
            test_size=(1 - self.train_split),
            seed=42
        )
        
        # Further split the test portion into val and test
        val_test_split = train_test_split['test'].train_test_split(
            test_size=0.5,  # Split the 20% into two 10% portions
            seed=42
        )
        
        self.datasets = {
            'train': train_test_split['train'],
            'validation': val_test_split['train'],
            'test': val_test_split['test']
        }
        
        print(f"Dataset splits:")
        print(f"  Train: {len(self.datasets['train'])} samples")
        print(f"  Validation: {len(self.datasets['validation'])} samples")
        print(f"  Test: {len(self.datasets['test'])} samples")
        
        return self.datasets
    
    def load_model_and_tokenizer(self):
        """Load BERT model and tokenizer"""
        print(f"Loading model and tokenizer: {self.model_name}")
        
        # Load BERT tokenizer - works normally
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True
        )
        
        # Load BERT model for MLM
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.model_name
        )
        
        print(f"Model loaded with {self.model.num_parameters():,} parameters")
        
    def tokenize_function(self, examples):
        """Tokenize the article text for BERT"""
        # Standard BERT tokenization
        return self.tokenizer(
            examples['article'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_special_tokens_mask=True
        )
    
    def prepare_datasets(self):
        """Tokenize all datasets"""
        print("Tokenizing datasets...")
        
        self.tokenized_datasets = {}
        for split_name, dataset in self.datasets.items():
            self.tokenized_datasets[split_name] = dataset.map(
                self.tokenize_function,
                batched=True,
                remove_columns=dataset.column_names,
                desc=f"Tokenizing {split_name}"
            )
        
        print("Tokenization complete")
        
    def create_data_collator(self):
        """Create data collator for MLM with 15% masking probability"""
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=0.15
        )
        
    def setup_training_args(
        self,
        num_epochs: int = 20,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        save_strategy: str = "steps",
        save_steps: int = 500,
        eval_steps: int = 500,
        logging_steps: int = 100,
    ):
        """Setup training arguments"""
        
        self.training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=2,  # Effective batch size = 8 * 2 = 16
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            
            # Evaluation and saving (use evaluation_strategy for transformers 4.34.0)
            evaluation_strategy="steps",
            eval_steps=eval_steps,
            save_strategy=save_strategy,
            save_steps=save_steps,
            save_total_limit=2,  # Only keep 2 best checkpoints
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # Logging
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=logging_steps,
            report_to="wandb",
            
            # Performance
            fp16=True,  # Use mixed precision for H100
            dataloader_num_workers=4,
            gradient_checkpointing=True,  # Save memory during training
            
            # Other
            seed=42,
            push_to_hub=False,
        )
        
        print(f"\nTraining configuration:")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Warmup ratio: {warmup_ratio}")
        print(f"  Weight decay: {weight_decay}")
        
    def train(self):
        """Run the training"""
        print("\nInitializing trainer...")
        
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.tokenized_datasets['train'],
            eval_dataset=self.tokenized_datasets['validation'],
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
        )
        
        print("Starting training...")
        train_result = self.trainer.train()
        
        print("\nTraining complete!")
        print(f"Training loss: {train_result.training_loss:.4f}")
        
        # Save the final model
        print(f"\nSaving best model to {self.output_dir}/best_model")
        self.trainer.save_model(f"{self.output_dir}/best_model")
        self.tokenizer.save_pretrained(f"{self.output_dir}/best_model")
        
        # Save metrics
        metrics = train_result.metrics
        self.trainer.save_metrics("train", metrics)
        
        return train_result
    
    def evaluate(self, split='validation'):
        """Evaluate on a specific split"""
        print(f"\nEvaluating on {split} set...")
        
        eval_results = self.trainer.evaluate(
            eval_dataset=self.tokenized_datasets[split]
        )
        
        print(f"{split.capitalize()} Results:")
        print(f"  Loss: {eval_results['eval_loss']:.4f}")
        print(f"  Perplexity: {np.exp(eval_results['eval_loss']):.4f}")
        
        self.trainer.save_metrics(split, eval_results)
        
        return eval_results


def main():
    """Main training pipeline"""
    
    # Initialize trainer
    mlm_trainer = BERTMLMTrainer(
        model_name="bert-base-multilingual-cased",
        dataset_name="jcblaise/fake_news_filipino",
        output_dir="./bert_tagalog_mlm",
        max_length=512,
        train_split=0.8,
        val_split=0.1,
    )
    
    # Load and prepare data
    mlm_trainer.load_and_prepare_data()
    
    # Load model and tokenizer
    mlm_trainer.load_model_and_tokenizer()
    
    # Tokenize datasets
    mlm_trainer.prepare_datasets()
    
    # Create data collator
    mlm_trainer.create_data_collator()
    
    # Setup training arguments
    mlm_trainer.setup_training_args(
        num_epochs=20,
        batch_size=8,  # Reduced from 16 to fit in GPU memory
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        save_steps=500,
        eval_steps=500,
        logging_steps=100,
    )
    
    # Train
    train_result = mlm_trainer.train()
    
    # Final evaluation on validation set
    mlm_trainer.evaluate(split='validation')
    
    # Optional: evaluate on test set
    print("\n" + "="*50)
    print("Final evaluation on test set:")
    mlm_trainer.evaluate(split='test')
    
    # Close wandb
    wandb.finish()
    
    print("\n" + "="*50)
    print("Training pipeline complete!")
    print(f"Best model saved to: {mlm_trainer.output_dir}/best_model")


if __name__ == "__main__":
    main()