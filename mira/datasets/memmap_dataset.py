#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
Memory-mapped numpy dataset for fast random access training.
No PKL parsing overhead - direct numpy array slicing.
"""

import os
import numpy as np
from mira.utils.log_util import logger
from mira.datasets.ts_dataset import TimeSeriesDataset


class MIRADataMemmapDataset(TimeSeriesDataset):
    """
    Fast memory-mapped dataset. All sequence data is stored in a single .npy file
    with offset metadata for O(1) random access.
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

        # Compute normalization
        if normalization_method == 'zero':
            sample_size = min(1000, self.num_sequences)
            indices = np.random.choice(self.num_sequences, sample_size, replace=False)
            all_vals = []
            for idx in indices:
                start, end = self.offsets[idx], self.offsets[idx + 1]
                all_vals.append(self.data[start:end].reshape(-1, 1))
            all_data = np.vstack(all_vals)
            self.mean = all_data.mean()
            self.std = all_data.std() if all_data.std() > 0 else 1.0
            logger.info(f"Computed normalization: mean={self.mean:.4f}, std={self.std:.4f}")
        else:
            self.mean = 0.0
            self.std = 1.0
            logger.info("No normalization applied.")

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, seq_idx):
        start = self.offsets[seq_idx]
        end = self.offsets[seq_idx + 1]
        sequence = self.data[start:end].copy()
        if self.std > 0:
            sequence = (sequence - self.mean) / self.std
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
    Ultra-fast window dataset. Precomputes absolute offsets into the flat data array.
    Each __getitem__ does exactly one numpy slice - no binary search, no intermediate dicts.
    """

    def __init__(self, dataset, context_length: int, prediction_length: int = 0, **kwargs):
        self.data = dataset.data
        self.mean = dataset.mean
        self.std = dataset.std
        self.offsets = dataset.offsets
        self.seq_lengths = dataset.seq_lengths

        self.window_size = context_length + prediction_length
        self.window_size_plus_one = self.window_size + 1

        # Precompute absolute start offset and length for each window
        # Each entry: (flat_data_start_offset, window_length)
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

        # Single memmap slice
        sequence = self.data[start:start + length].copy().astype(np.float32)
        if self.std > 0:
            sequence = (sequence - self.mean) / self.std

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
