#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
Classification trainer for MIRA.
Extends MIRATrainer with compute_metrics for AUC-ROC, Precision, Recall, F1.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from mira.trainer.hf_trainer import MIRATrainer


class MIRAClassificationTrainer(MIRATrainer):
    """Trainer for classification tasks with medical evaluation metrics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred

        # Binary classification: sigmoid of positive-class logit (last column)
        if logits.shape[-1] == 2:
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits[:, -1], -500, 500)))
            preds = (probs >= 0.5).astype(int)
        else:
            probs = self._softmax(logits)
            preds = np.argmax(probs, axis=1)
            probs = probs[:, 1] if probs.shape[-1] > 1 else probs.flatten()

        unique_labels = np.unique(labels)
        auc = roc_auc_score(labels, probs) if len(unique_labels) > 1 else 0.0

        return {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0),
            "auc_roc": auc,
        }

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def _softmax(x):
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)
