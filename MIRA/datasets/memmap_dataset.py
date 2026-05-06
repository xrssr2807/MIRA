#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
Memory-mapped numpy dataset for fast random access training.
Global z-score normalization: one mean/std computed from ALL training data.
"""

import os
import pickle
import numpy as np
from MIRA.utils.log_util import logger
from MIRA.datasets.ts_dataset import TimeSeriesDataset


class MIRADataMemmapDataset(TimeSeriesDataset):
    """
    Fast memory-mapped dataset. All sequence data is stored in a single .npy file
    with offset metadata for O(1) random access.
    Global z-score normalization: one mean/std for the entire dataset.
    """

    def __init__(
        self,
        data_dir,
        normalization_method='zero',
    ):
        data_path = os.path.join(data_dir, "ppg_data.npy")
        offsets_path = os.path.join(data_dir, "ppg_offsets.npy")

        if not os.path.exists(data_path) or not os.path.exists(offsets_path):
            raise FileNotFoundError(f"Missing npy files in {data_dir}")

        logger.info(f"Loading memmap dataset from {data_dir}...")

        # Load offsets
        self.offsets = np.load(offsets_path)
        self.num_sequences = len(self.offsets) - 1
        self.seq_lengths = np.diff(self.offsets).astype(np.int64)

        # Load data as memmap (lazy, no memory load)
        total_len = self.offsets[-1]
        self.data = np.memmap(data_path, dtype='float32', mode='r', shape=(total_len,))

        logger.info(f"Loaded {self.num_sequences} sequences, {total_len} total points.")

        # Compute GLOBAL mean/std from all training data
        if normalization_method == 'zero':
            logger.info("Computing GLOBAL z-score stats from all data...")
            # Sample a subset for efficiency (computing on 5.3GB is slow)
            sample_size = min(total_len, 50_000_000)
            if sample_size < total_len:
                rng = np.random.RandomState(42)
                indices = rng.choice(total_len, sample_size, replace=False)
                sampled = self.data[indices]
                self.global_mean = float(sampled.mean())
                self.global_std = float(sampled.std())
            else:
                self.global_mean = float(self.data.mean())
                self.global_std = float(self.data.std())

            if self.global_std == 0:
                self.global_std = 1.0
            logger.info(f"Global normalization: mean={self.global_mean:.4f}, std={self.global_std:.4f}")

            # Save to PKL for downstream tasks
            norm_stats_path = os.path.join(data_dir, "norm_stats.pkl")
            with open(norm_stats_path, 'wb') as f:
                pickle.dump({"global_mean": self.global_mean, "global_std": self.global_std}, f)
            logger.info(f"Saved global norm stats to {norm_stats_path}")
        else:
            self.global_mean = 0.0
            self.global_std = 1.0
            logger.info("No normalization applied.")

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, seq_idx):
        start = self.offsets[seq_idx]
        end = self.offsets[seq_idx + 1]
        sequence = self.data[start:end].copy()
        sequence = (sequence - self.global_mean) / self.global_std
        n = len(sequence)
        time = np.arange(n, dtype=np.float32) * 10.0
        mask = np.ones(n, dtype=np.int32)
        return {"sequence": sequence, "time": time, "mask": mask}

    def get_num_tokens(self):
        return int(self.offsets[-1])

    def get_sequence_length_by_idx(self, seq_idx):
        return self.seq_lengths[seq_idx]

    def get_time_normalizer(self):
        return None


class FlatWindowDataset:
    """
    Window dataset with GLOBAL z-score normalization.
    Slices from raw memmap, then applies the same global mean/std.
    """

    def __init__(self, dataset, context_length: int, prediction_length: int = 0, **kwargs):
        self.data = dataset.data
        self.offsets = dataset.offsets
        self.seq_lengths = dataset.seq_lengths
        # Global normalization params
        self.global_mean = dataset.global_mean
        self.global_std = dataset.global_std

        self.window_size = context_length + prediction_length
        self.window_size_plus_one = self.window_size + 1

        # Precompute window metadata: start offset, length
        logger.info("Precomputing window offsets...")
        window_starts = []
        window_lens = []
        for seq_idx in range(len(dataset)):
            seq_len = dataset.get_sequence_length_by_idx(seq_idx)
            seq_start = dataset.offsets[seq_idx]
            n_windows = seq_len // self.window_size if seq_len >= self.window_size else 0
            for w in range(n_windows):
                w_start = seq_start + w * self.window_size
                w_len = min(self.window_size_plus_one, seq_len - w * self.window_size)
                if w_len < 2:
                    break
                window_starts.append(w_start)
                window_lens.append(w_len)

        self.window_starts = np.array(window_starts, dtype=np.int64)
        self.window_lens = np.array(window_lens, dtype=np.int32)
        logger.info(f"Created {len(self)} windows from {len(dataset)} sequences.")

    def __len__(self):
        return len(self.window_starts)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx):
        start = self.window_starts[idx]
        length = int(self.window_lens[idx])

        # Slice from raw memmap
        sequence = self.data[start:start + length].copy().astype(np.float32)

        # Apply GLOBAL z-score normalization
        sequence = (sequence - self.global_mean) / self.global_std

        # Loss mask and padding
        loss_mask = np.ones(length - 1, dtype=np.int32)
        time = np.arange(length, dtype=np.float32) * 10.0

        n_pad = self.window_size_plus_one - length
        if n_pad > 0:
            sequence = np.pad(sequence, (0, n_pad), 'constant', constant_values=0)
            time = np.pad(time, (0, n_pad), 'constant', constant_values=0)
            loss_mask = np.pad(loss_mask, (0, n_pad), 'constant', constant_values=0)

        return {
            'input_ids': sequence[:-1],
            'labels': sequence[1:],
            'loss_masks': loss_mask,
            'time_values': time[:-1],
        }
