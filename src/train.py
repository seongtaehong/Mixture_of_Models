#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestrator Model Training Script

This script trains a classification model (orchestrator) that learns to predict 
which model will perform best for a given input question. The orchestrator is 
trained on labeled data where each sample indicates the best-performing model.

Usage:
    python train.py --model_name "answerdotai/ModernBERT-base" \
                    --data_path "./labeled_data/training.jsonl" \
                    --output_dir "./ckpt/modernbert" \
                    --gpu "0"
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    set_seed
)
from datasets import Dataset
import evaluate
import logging

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def parse_args():
    """
    Parse command line arguments for training configuration.
    
    Returns:
        argparse.Namespace: Parsed arguments containing all training parameters
    """
    parser = argparse.ArgumentParser(description="Train orchestrator model on JSONL dataset for model selection.")
    
    # Model configuration
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/deberta-v3-base",
        help="Name or path of the base model (e.g., microsoft/deberta-v3-base)"
    )
    
    # Data configuration
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to JSONL file containing labeled training data"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.05,
        help="Proportion of data to use for validation (0.0 ~ 1.0)"
    )
    
    # Training configuration
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Training batch size"
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=64,
        help="Evaluation batch size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for optimization"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps"
    )
    
    # Model configuration
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum token length for input sequences"
    )
    
    # System configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./model_output",
        help="Directory to save model checkpoints and outputs"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--gpu", 
        type=str,
        default="7",
        help="GPU device number to use (e.g., '0,1,2,3' for multiple GPUs)"
    )
    
    return parser.parse_args()


def load_data(data_path):
    """
    Load training data from JSONL file.
    
    Each line in the JSONL file should contain a JSON object with at least
    'question' and 'label' fields, where 'label' indicates the best model.
    
    Args:
        data_path (str): Path to the JSONL file containing training data
        
    Returns:
        list: List of dictionaries, each containing a training sample
        
    Raises:
        Exception: If file cannot be read or parsed
    """
    logger.info(f"Loading data from: {data_path}")
    data = []
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
                
        logger.info(f"Data loading completed: {len(data)} samples")
        return data
    except Exception as e:
        logger.error(f"Error occurred while loading data: {e}")
        raise


def prepare_datasets(data, test_size=0.2, seed=42):
    """
    Prepare training and validation datasets from loaded data.
    
    This function converts the data to pandas DataFrame, creates label mappings,
    and splits the data into training and validation sets.
    
    Args:
        data (list): List of training samples
        test_size (float): Proportion of data to use for validation
        seed (int): Random seed for reproducible splits
        
    Returns:
        tuple: (train_df, val_df, id2label, label2id)
            - train_df: Training DataFrame
            - val_df: Validation DataFrame  
            - id2label: Mapping from label IDs to label names
            - label2id: Mapping from label names to label IDs
    """
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Create label mappings
    unique_labels = df['label'].unique()
    id2label = {i: label for i, label in enumerate(unique_labels)}
    label2id = {label: i for i, label in enumerate(unique_labels)}
    
    # Convert labels to integers
    df['labels'] = df['label'].map(label2id)
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)
    
    logger.info(f"Data split completed: {len(train_df)} training, {len(val_df)} validation")
    logger.info(f"Number of unique labels: {len(unique_labels)}")
    
    return train_df, val_df, id2label, label2id


def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for model performance.
    
    Args:
        eval_pred (tuple): Tuple containing (predictions, labels)
            - predictions: Model predictions as logits
            - labels: True labels
            
    Returns:
        dict: Dictionary containing computed metrics (accuracy)
    """
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def train_model(args):
    """
    Main training function that orchestrates the entire training process.
    
    This function handles:
    - GPU configuration and seed setting
    - Directory creation for outputs and logs
    - Data loading and preprocessing
    - Model and tokenizer initialization
    - Dataset preparation and tokenization
    - Training argument configuration
    - Model training and evaluation
    - Model saving
    
    Args:
        args (argparse.Namespace): Parsed command line arguments
        
    Returns:
        dict: Evaluation results from the trained model
    """
    # System setup
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_seed(args.seed)
    
    # Create output directories
    if not os.path.exists(args.output_dir):
        logger.info(f"Creating output directory: {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)
    
    log_dir = f"{args.output_dir}/logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # Data preparation
    data = load_data(args.data_path)
    train_df, val_df, id2label, label2id = prepare_datasets(data, args.test_size, args.seed)
    
    # Model initialization
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id
    )
    
    # Tokenization function
    def tokenize_function(examples):
        """Tokenize input questions with padding and truncation."""
        return tokenizer(examples["question"], padding="max_length", truncation=True, max_length=args.max_length)
    
    # Dataset preparation
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    # Remove unnecessary columns
    columns_to_remove = [col for col in train_dataset.column_names 
                         if col not in ["input_ids", "attention_mask", "labels"]]
    train_dataset = train_dataset.remove_columns(columns_to_remove)
    val_dataset = val_dataset.remove_columns(columns_to_remove)
    
    # Set dataset format
    train_dataset.set_format("torch")
    val_dataset.set_format("torch")
    
    # Training configuration
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=0.01,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=0.9,
        eval_steps=0.9,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="none"
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Training process
    logger.info("Starting model training")
    trainer.train()
    
    # Model evaluation
    logger.info("Evaluating model")
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")
    
    # Save final model
    logger.info(f"Saving final model to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    return eval_results


def main():
    """
    Main entry point of the training script.
    
    Parses command line arguments and initiates the training process.
    """
    args = parse_args()
    train_model(args)


if __name__ == "__main__":
    main()


# Example usage commands:
# python train.py --model_name "microsoft/deberta-v3-base" --data_path "./labeled_data/2_8_10.jsonl" --output_dir "./ckpt/deberta" --gpu "0" > deberta.log  2>&1 & 
# python train.py --model_name "answerdotai/ModernBERT-base" --data_path "./labeled_data/2_3_10.jsonl" --output_dir "./ckpt/modern" --gpu "0" > modernbert.log  2>&1 & 
