#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MIRA Signal Classification Evaluation Script.

Evaluates a trained classifier on test data, computing:
- AUC-ROC: Overall discrimination ability
- Precision: Ratio of true positives among predicted positives (false positive rate)
- Recall (Sensitivity): Ratio of true positives found (false negative rate)
- F1 Score: Harmonic mean of precision and recall
- Confusion matrix

Usage:
    python eval_classifier.py --model ppg_output/classifier_v1 --data pkg_full
    python eval_classifier.py --model ppg_output/classifier_v1 --data pkg_full --output results/classifier_eval
"""
import os
import sys
import json
import argparse
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader

# Set non-interactive backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix, roc_curve, precision_recall_curve
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mira.models.modeling_mira import MIRAForClassification, MIRAConfig
from mira.datasets.classification_dataset import PPGClassificationDataset, classification_collate_fn


def evaluate(model, dataloader, device):
    """Run inference and collect all logits and labels."""
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            input_ids = batch["input_ids"].to(device)
            time_values = batch["time_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                time_values=time_values,
                attention_mask=attention_mask,
                labels=None,
                return_dict=True,
            )
            all_logits.append(outputs.logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_logits, all_labels


def compute_metrics(logits, labels):
    """Compute classification metrics."""
    if logits.shape[-1] == 2:
        # Binary: use sigmoid of the positive-class logit (last column)
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits[:, -1], -500, 500)))
        preds = (probs >= 0.5).astype(int)
    else:
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500))).flatten()
        preds = (probs >= 0.5).astype(int)

    unique = np.unique(labels)
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "auc_roc": float(roc_auc_score(labels, pos_probs)) if len(unique) > 1 else 0.0,
        "n_samples": int(len(labels)),
        "n_positive": int((labels == 1).sum()),
        "n_negative": int((labels == 0).sum()),
    }

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    metrics["confusion_matrix"] = cm.tolist()

    return metrics, preds, pos_probs


def plot_roc_curve(labels, probs, output_dir):
    """Plot and save ROC curve."""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curve to {output_dir}/roc_curve.png")


def plot_pr_curve(labels, probs, metrics_dict, output_dir):
    """Plot and save Precision-Recall curve."""
    precision, recall, thresholds = precision_recall_curve(labels, probs)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='darkblue', lw=2,
            label=f'PR curve (AP={metrics_dict["precision"]:.4f}, R={metrics_dict["recall"]:.4f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall (Sensitivity)')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'pr_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved PR curve to {output_dir}/pr_curve.png")


def plot_confusion_matrix(cm, output_dir):
    """Plot and save confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Normal', 'Abnormal'])
    ax.set_yticklabels(['Normal', 'Abnormal'])

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]}',
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {output_dir}/confusion_matrix.png")


def main():
    parser = argparse.ArgumentParser(description='Evaluate MIRA signal classifier')
    parser.add_argument('--model', type=str, required=True, help='Path to trained classifier model')
    parser.add_argument('--data', type=str, required=True, help='Path to test data (PKL files)')
    parser.add_argument('--output', type=str, default='eval_results', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output}\n")

    # Load model
    print("Loading model...")
    config = MIRAConfig.from_pretrained(args.model)
    model = MIRAForClassification.from_pretrained(
        args.model,
        config=config,
        num_classes=args.num_classes,
        attn_implementation='eager',
    ).to(device)
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M\n")

    # Load dataset
    print("Loading dataset...")
    dataset = PPGClassificationDataset(args.data, normalization_method="zero")
    print(f"Total samples: {len(dataset)}\n")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=classification_collate_fn,
    )

    # Run inference
    print("Running inference...")
    logits, labels = evaluate(model, dataloader, device)
    print(f"Evaluated {len(labels)} samples\n")

    # Compute metrics
    metrics, preds, probs = compute_metrics(logits, labels)

    # Print results
    print("=" * 50)
    print("  CLASSIFICATION EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Samples: {metrics['n_samples']} (Positive: {metrics['n_positive']}, Negative: {metrics['n_negative']})")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    {metrics['confusion_matrix']}")
    print("=" * 50)

    # Save results
    os.makedirs(args.output, exist_ok=True)

    with open(os.path.join(args.output, "eval_results.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {args.output}/eval_results.json")

    # Save raw predictions for further analysis
    np.savez(
        os.path.join(args.output, "predictions.npz"),
        logits=logits,
        labels=labels,
        probs=probs,
        preds=preds,
    )
    print(f"Saved raw predictions to {args.output}/predictions.npz")

    # Plot visualizations
    plot_roc_curve(labels, probs, args.output)
    plot_pr_curve(labels, probs, metrics, args.output)
    plot_confusion_matrix(np.array(metrics["confusion_matrix"]), args.output)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
