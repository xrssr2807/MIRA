#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
Classification dataset for MIRA.
Loads PPG sequences with labels from PKL files.

Expected PKL format (one dict per file):
    {
        'uid': 'MPA_181220181216',
        'data': ndarray (1, 1000), float16,
        'sampling_rate': 100,
        'label': [{'class': 0}]
    }
"""

import os
import glob
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from mira.utils.log_util import logger


def extract_label(label_value):
    """Extract integer class label from various formats."""
    if isinstance(label_value, int):
        return label_value
    if isinstance(label_value, (list, tuple)) and len(label_value) > 0:
        item = label_value[0]
        if isinstance(item, dict):
            return int(item.get("class", 0))
        return int(item)
    return 0


class PPGClassificationDataset(Dataset):
    """Lazy-loading PPG classification dataset.

    Each PKL file = one sample (dict with 'data' and 'label' keys).
    """

    def __init__(self, data_path, normalization_method="zero"):
        if not os.path.exists(data_path):
            raise ValueError(f"Invalid data path: {data_path}")

        self.normalization_method = normalization_method

        if os.path.isfile(data_path) and data_path.endswith('.pkl'):
            pkl_files = [data_path]
        elif os.path.isdir(data_path):
            pkl_files = sorted(glob.glob(os.path.join(data_path, "*.pkl")))
        else:
            raise ValueError(f"Expected a .pkl file or directory of .pkl files, got: {data_path}")

        if not pkl_files:
            raise ValueError(f"No .pkl files found in: {data_path}")

        self.pkl_files = pkl_files
        self.index = []  # (file_path, seq_len, label)
        self.labels = []
        self.raw_labels = []  # store original labels for reference

        logger.info(f"Indexing {len(pkl_files)} PKL files for classification...")
        for fp in pkl_files:
            with open(fp, 'rb') as f:
                item = pickle.load(f)
            seq = item["data"]
            raw_label = extract_label(item.get("label", 0))
            # Binary: 0=normal, 1=abnormal (any non-zero class)
            label = 0 if raw_label == 0 else 1

            if isinstance(seq, np.ndarray) and seq.ndim == 2:
                seq = seq[0]
            seq = np.array(seq, dtype=np.float32)
            label = int(label)

            self.index.append((fp, len(seq), label))
            self.labels.append(label)
            self.raw_labels.append(raw_label)

        self.num_sequences = len(self.index)
        unique, counts = np.unique(self.labels, return_counts=True)
        logger.info(f"Indexed {self.num_sequences} samples. Label distribution (binary): {dict(zip(unique.tolist(), counts.tolist()))}")

        # Fit normalizer on all sequences
        self.data_normalizer = None
        self._fit_normalizer()

    def _fit_normalizer(self):
        if self.normalization_method not in ("zero", "standard", "minmax"):
            return
        logger.info("Fitting data normalizer on all sequences...")
        all_vals = []
        for fp, seq_len, label in self.index:
            with open(fp, 'rb') as f:
                item = pickle.load(f)
            seq = item["data"]
            if isinstance(seq, np.ndarray) and seq.ndim == 2:
                seq = seq[0]
            all_vals.append(seq.reshape(-1, 1))

        if not all_vals:
            logger.warning("No valid data found for normalization.")
            return

        all_data = np.vstack(all_vals)
        if self.normalization_method == "standard":
            self.data_normalizer = StandardScaler()
            self.data_normalizer.fit(all_data)
            logger.info(f"Fitted StandardScaler on {all_data.shape[0]} values.")
        elif self.normalization_method == "zero":
            self.data_mean = float(all_data.mean())
            self.data_std = float(all_data.std()) + 1e-8
            logger.info(f"Zero-mean normalization: mean={self.data_mean:.4f}, std={self.data_std:.4f}")

    def _normalize(self, seq):
        if self.data_normalizer is not None:
            return self.data_normalizer.transform(seq.reshape(-1, 1)).reshape(-1)
        elif self.normalization_method == "zero":
            return (seq - self.data_mean) / self.data_std
        return seq

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        fp, seq_len, label = self.index[idx]
        with open(fp, 'rb') as f:
            item = pickle.load(f)
        seq = item["data"]
        if isinstance(seq, np.ndarray) and seq.ndim == 2:
            seq = seq[0]
        seq = np.array(seq, dtype=np.float32)
        seq = self._normalize(seq)

        time = np.arange(len(seq), dtype=np.float32)

        return {
            "input_ids": seq,
            "time_values": time,
            "labels": int(label),
        }

    def get_sequence_length_by_idx(self, idx):
        return self.index[idx][1]


def classification_collate_fn(batch, max_length=None):
    """Pad variable-length sequences and create attention masks for classification."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    max_len = max(len(b["input_ids"]) for b in batch)
    if max_length is not None and max_len > max_length:
        max_len = max_length

    input_ids_list = []
    time_values_list = []
    attention_mask_list = []
    labels_list = []

    for item in batch:
        L = len(item["input_ids"])
        if max_length is not None and L > max_length:
            seq = item["input_ids"][:max_length]
            time = item["time_values"][:max_length]
            mask = np.ones(max_length, dtype=np.int64)
            pad_len = 0
        else:
            seq = item["input_ids"]
            time = item["time_values"]
            pad_len = max_len - L
            pad_time = float(item["time_values"][-1]) if L > 0 else 0.0
            seq = np.pad(seq, (0, pad_len), constant_values=0)
            time = np.pad(time, (0, pad_len), constant_values=pad_time)
            mask = np.pad(np.ones(L, dtype=np.int64), (0, pad_len), constant_values=0)

        input_ids_list.append(seq)
        time_values_list.append(time)
        attention_mask_list.append(mask)
        labels_list.append(item["labels"])

    return {
        "input_ids": torch.tensor(np.stack(input_ids_list), dtype=torch.float32).unsqueeze(-1),  # [B, L, 1]
        "time_values": torch.tensor(np.stack(time_values_list), dtype=torch.float32),            # [B, L]
        "attention_mask": torch.tensor(np.stack(attention_mask_list), dtype=torch.long),          # [B, L]
        "labels": torch.tensor(labels_list, dtype=torch.long),                                    # [B]
    }
