#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MIRA Signal Classification Training Script.

Fine-tunes MIRA for binary signal classification (e.g., AFib detection, disease screening).
Trains with 80% data, evaluates on 20% with AUC-ROC, Precision, Recall metrics.

Usage:
    python train_classifier.py
"""
import os
import sys
from functools import partial
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MIRA.models.modeling_mira import MIRAForClassification, MIRAConfig
from MIRA.datasets.classification_dataset import PPGClassificationDataset, classification_collate_fn
from MIRA.trainer.classification_trainer import MIRAClassificationTrainer
from MIRA.trainer.hf_trainer import MIRATrainingArguments

# ============================================================
#  Configuration — modify these for your downstream task
# ============================================================

# Pre-trained model checkpoint directory (contains config.json + model.safetensors)
MODEL_PATH = "/root/model"

# Classification training data path (PKL files with 'data' and 'label' keys)
DATA_PATH = "/root/processed_dataset/data"

# Output directory for saved model and logs
OUTPUT_PATH = "ppg_output/classifier_v1"

# Number of classes (2 for binary: normal vs abnormal)
NUM_CLASSES = 2

# Set True to freeze backbone and only train classification head
FREEZE_BACKBONE = False

# Max sequence length (truncates longer signals)
MAX_LENGTH = 512

# Training hyperparameters
LEARNING_RATE = 5e-5
NUM_EPOCHS = 10
MICRO_BATCH_SIZE = 64
GRADIENT_ACCUMULATION = 4

# Train/eval split ratio
TRAIN_RATIO = 0.8

# Random seed
SEED = 42

# ============================================================


def main():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Backbone: {MODEL_PATH}")
    print(f"Data: {DATA_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Classes: {NUM_CLASSES}, Freeze backbone: {FREEZE_BACKBONE}")
    print(f"Batch size: {MICRO_BATCH_SIZE} x {GRADIENT_ACCUMULATION} = {MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION}")

    # --- Load dataset ---
    print("\nLoading dataset...")
    full_dataset = PPGClassificationDataset(DATA_PATH, normalization_method="zero")
    print(f"Total samples: {len(full_dataset)}")

    # Sequential 80/20 split: first 80% for training, last 20% for evaluation
    train_size = int(len(full_dataset) * TRAIN_RATIO)
    train_dataset = torch.utils.data.Subset(full_dataset, range(train_size))
    eval_dataset = torch.utils.data.Subset(full_dataset, range(train_size, len(full_dataset)))
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # --- Load model ---
    print("\nLoading model...")
    config = MIRAConfig.from_pretrained(MODEL_PATH)
    model = MIRAForClassification(config, num_classes=NUM_CLASSES)

    # Load pre-trained backbone weights
    from MIRA.models.modeling_mira import MIRAForPrediction
    backbone = MIRAForPrediction.from_pretrained(MODEL_PATH, torch_dtype=torch.float32, attn_implementation='eager')
    model.model.load_state_dict(backbone.model.state_dict(), strict=True)
    del backbone
    print(f"Backbone weights loaded from {MODEL_PATH}")

    # Optionally freeze backbone
    if FREEZE_BACKBONE:
        for param in model.model.parameters():
            param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Frozen backbone. Trainable params: {trainable / 1e6:.2f}M")
    else:
        total = sum(p.numel() for p in model.parameters())
        print(f"Full fine-tuning. Total params: {total / 1e6:.2f}M")

    model = model.to(device)
    model = torch.compile(model, dynamic=True)

    # --- Training args ---
    training_args = MIRATrainingArguments(
        output_dir=OUTPUT_PATH,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        per_device_eval_batch_size=MICRO_BATCH_SIZE * 2,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        gradient_checkpointing=True,
        bf16=torch.cuda.is_available(),
        fp16=False,
        eval_strategy="epoch",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        save_only_model=True,
        weight_decay=0.01,
        max_grad_norm=1.0,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        seed=SEED,
        data_seed=SEED,
        ddp_find_unused_parameters=True,
        load_best_model_at_end=True,
        metric_for_best_model="auc_roc",
        greater_is_better=True,
    )

    # --- Train ---
    trainer = MIRAClassificationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=partial(classification_collate_fn, max_length=MAX_LENGTH),
    )

    print("\nStarting training...")
    trainer.train()

    # Save final model
    trainer.save_model(OUTPUT_PATH)
    print(f"\nModel saved to {OUTPUT_PATH}")

    # Print final eval metrics
    metrics = trainer.evaluate()
    print("\n" + "=" * 50)
    print("  FINAL EVALUATION METRICS")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
