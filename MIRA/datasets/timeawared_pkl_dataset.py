#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
Lazy time-aware PKL dataset for MIRA.
Only loads PKL files on demand during training, not all at once.
"""

import os
import glob
import pickle
import random
import gc
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from mira.utils.log_util import logger
from mira.datasets.ts_dataset import TimeSeriesDataset


def quantize_time(times, initial_resolution=1.0, min_resolution=1e-8, shrink_factor=10, jitter_eps=1e-8, max_iterations=20):
    times = np.array(times, dtype=np.float64)
    resolution = initial_resolution
    for it in range(max_iterations):
        quantized = np.round(times / resolution) * resolution
        if len(np.unique(quantized)) == len(times):
            return quantized.astype(np.float32)
        resolution = max(resolution / shrink_factor, min_resolution)
    quantized = np.round(times / resolution) * resolution
    unique_quantized = []
    last_value = None
    max_abs_time = np.max(np.abs(times)) if len(times) > 0 else 1.0
    current_eps = max(jitter_eps, np.finfo(np.float32).eps * max_abs_time)
    current_eps = min(current_eps, max_abs_time * 1e-6)
    for q in quantized:
        if last_value is not None and q <= last_value:
            q = last_value + current_eps
        unique_quantized.append(q)
        last_value = q
    times_value = np.array(unique_quantized, dtype=np.float32)
    _, indices = np.unique(times_value, return_index=True)
    duplicates = np.setdiff1d(np.arange(len(times_value)), indices)
    for idx in duplicates:
        times_value[idx] += current_eps
    return times_value


class TimeAwarePKLDataset(TimeSeriesDataset):
    """
    Lazy-loading time-aware PKL dataset.
    Does NOT load all data into memory. Instead, it reads PKL files on demand.
    """

    def __init__(
        self,
        data_path,
        time_normalization="none",
        quantize_resolution=None,
        auto_quantize=False,
        sample_size=1000,
        data_normalizer=MinMaxScaler(),
    ):
        if not os.path.exists(data_path):
            raise ValueError(f"Invalid data path: {data_path}")

        self.time_normalization = time_normalization.lower() if isinstance(time_normalization, str) else None
        self.time_normalizer = None
        self.quantize_resolution = quantize_resolution
        self.auto_quantize = auto_quantize
        self.data_normalizer = data_normalizer
        self.FS = 100
        self.INTERVAL_MS = 1000.0 / self.FS

        # Build index: list of (pkl_file_path, item_index_in_file, channel_index, seq_len)
        if os.path.isfile(data_path) and data_path.endswith('.pkl'):
            pkl_files = [data_path]
        elif os.path.isdir(data_path):
            pkl_files = sorted(glob.glob(os.path.join(data_path, "*.pkl")))
        else:
            raise ValueError(f"Expected a .pkl file or directory of .pkl files, got: {data_path}")

        if not pkl_files:
            raise ValueError(f"No .pkl files found in: {data_path}")

        logger.info(f"Indexing {len(pkl_files)} pkl files...")
        self.pkl_files = pkl_files
        self.index = []  # (file_idx, item_idx, channel_idx, seq_len)
        self.file_loaded = {}  # file_idx -> loaded data (LRU cache)
        self.max_cached = 2  # keep at most 2 files in memory

        for file_idx, pkl_path in enumerate(pkl_files):
            with open(pkl_path, 'rb') as f:
                items = pickle.load(f)
            for item_idx, item in enumerate(items):
                seq = item["data"]
                if isinstance(seq, np.ndarray) and seq.ndim == 2:
                    n_channels = seq.shape[0]
                    seq_len = seq.shape[1]
                    for ch in range(n_channels):
                        self.index.append((file_idx, item_idx, ch, seq_len))
                elif isinstance(seq, np.ndarray) and seq.ndim == 1:
                    self.index.append((file_idx, item_idx, 0, len(seq)))

        self.num_sequences = len(self.index)
        logger.info(f"Indexed {self.num_sequences} sequences from {len(pkl_files)} pkl files.")

        # Cache sequence lengths for fast access
        self.seq_lengths = [entry[3] for entry in self.index]

        # Fit normalizers on a sample
        self._fit_data_normalizer(sample_size)
        self._fit_time_normalizer(sample_size)

        logger.info("Finished loading meta info for PKL data.")

    def _get_data(self, file_idx):
        """Load a PKL file into cache."""
        if file_idx not in self.file_loaded:
            with open(self.pkl_files[file_idx], 'rb') as f:
                self.file_loaded[file_idx] = pickle.load(f)
            # Evict old entries
            while len(self.file_loaded) > self.max_cached:
                oldest = next(iter(self.file_loaded))
                del self.file_loaded[oldest]
                gc.collect()
        return self.file_loaded[file_idx]

    def _get_channel_data(self, file_idx, item_idx, channel_idx):
        """Extract a single channel from a cached PKL file."""
        items = self._get_data(file_idx)
        seq = items[item_idx]["data"]
        if isinstance(seq, np.ndarray) and seq.ndim == 2:
            return seq[channel_idx].astype(np.float32)
        elif isinstance(seq, np.ndarray) and seq.ndim == 1:
            return seq.astype(np.float32)
        return None

    def _fit_data_normalizer(self, sample_size=1000):
        all_vals = []
        sample_size = min(sample_size, self.num_sequences)
        indices = random.sample(range(self.num_sequences), sample_size)

        for idx in indices:
            file_idx, item_idx, ch_idx, _ = self.index[idx]
            seq = self._get_channel_data(file_idx, item_idx, ch_idx)
            if seq is not None and len(seq) > 0:
                all_vals.append(seq.reshape(-1, 1))

        if not all_vals:
            logger.warning("No valid sequence data found. Disabling value normalization.")
            self.data_normalizer = None
            return

        all_data = np.vstack(all_vals)
        try:
            self.data_normalizer.fit(all_data)
            logger.info(f"Fitted data normalizer on {all_data.shape[0]} values.")
        except Exception as e:
            logger.error(f"Error fitting data normalizer: {e}. Normalization disabled.")
            self.data_normalizer = None

    def _fit_time_normalizer(self, sample_size=1000):
        all_times, all_deltas = [], []
        sample_size = min(sample_size, self.num_sequences)
        indices = random.sample(range(self.num_sequences), sample_size)

        for idx in indices:
            seq_len = self.index[idx][3]
            time = np.arange(seq_len, dtype=np.float64) * self.INTERVAL_MS
            if len(time) >= 2:
                all_times.append(time)
                deltas = np.diff(time)
                deltas = deltas[deltas > 0]
                if len(deltas) > 0:
                    all_deltas.append(deltas)

        if len(all_times) == 0:
            logger.warning("[TimeNorm] No valid times found; skip normalization/quantization.")
            return

        flat_times = np.concatenate(all_times)

        if self.time_normalization in ["standard", "std"]:
            self.time_normalizer = StandardScaler()
            self.time_normalizer.fit(flat_times.reshape(-1, 1))
            logger.info("[TimeNorm] Fitted StandardScaler for time.")
        elif self.time_normalization == "minmax":
            self.time_normalizer = MinMaxScaler(feature_range=(0, 1))
            self.time_normalizer.fit(flat_times.reshape(-1, 1))
            logger.info("[TimeNorm] Fitted MinMaxScaler for time.")
        else:
            self.time_normalizer = None
            logger.info("[TimeNorm] Time normalization disabled (using raw timestamps).")

        if self.auto_quantize:
            if len(all_deltas) > 0:
                median_delta = np.median(np.concatenate(all_deltas))
                self.quantize_resolution = max(median_delta, 1e-9)
                logger.info(f"[Quantize] Inferred resolution: {self.quantize_resolution:.6f}")
            else:
                self.quantize_resolution = 1.0
                logger.warning("[Quantize] No deltas found, defaulting to resolution=1.0.")

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, seq_idx):
        file_idx, item_idx, ch_idx, seq_len = self.index[seq_idx]
        sequence = self._get_channel_data(file_idx, item_idx, ch_idx)
        n = len(sequence)

        time = np.arange(n, dtype=np.float64) * self.INTERVAL_MS
        mask = np.ones(n, dtype=np.int32)

        # Quantization
        if self.quantize_resolution is not None:
            time = quantize_time(time, initial_resolution=self.quantize_resolution)

        # Optional time normalization
        if self.time_normalizer is not None:
            time = self.time_normalizer.transform(time.reshape(-1, 1)).flatten()

        # Sequence normalization
        if self.data_normalizer is not None:
            sequence = self.data_normalizer.transform(sequence.reshape(-1, 1)).reshape(-1)

        return {
            "sequence": sequence.astype(np.float32),
            "time": time.astype(np.float32),
            "mask": mask.astype(np.int32),
        }

    def get_num_tokens(self):
        return sum(self.seq_lengths)

    def get_sequence_length_by_idx(self, seq_idx):
        return self.seq_lengths[seq_idx]

    def get_time_normalizer(self):
        return self.time_normalizer


class MIRAWindowPKLDataset:
    """
    Lightweight window dataset for PKL data. Does NOT precompute all windows.
    Generates windows on-the-fly to avoid OOM.
    Each sequence of length L produces floor(L / context_length) windows.
    """

    def __init__(self, dataset, context_length: int, prediction_length: int = 0, **kwargs):
        self.dataset = dataset
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.window_size = context_length + prediction_length
        self.window_size_plus_one = self.window_size + 1

        # Precompute cumulative window offsets (small memory footprint)
        # For each sequence, compute how many windows it produces
        self.cum_windows = [0]  # cumulative count of windows up to each sequence
        for seq_idx in range(len(dataset)):
            n_points = dataset.get_sequence_length_by_idx(seq_idx)
            n_windows = n_points // self.window_size if n_points >= self.window_size else 0
            self.cum_windows.append(self.cum_windows[-1] + n_windows)

        self.total_windows = self.cum_windows[-1]

    def __len__(self):
        return self.total_windows

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, window_idx):
        if window_idx < 0 or window_idx >= self.total_windows:
            raise IndexError(f"Window index {window_idx} out of range [0, {self.total_windows})")

        # Find which sequence this window belongs to (binary search)
        seq_idx = self._find_sequence(window_idx)
        # How many windows before this sequence
        windows_before = self.cum_windows[seq_idx]
        # Window index within this sequence
        window_in_seq = window_idx - windows_before

        offset = window_in_seq * self.window_size
        seq = self.dataset[seq_idx]
        sequence = seq['sequence'][offset: offset + self.window_size_plus_one]
        time = seq['time'][offset: offset + self.window_size_plus_one]

        # Handle padding if needed
        loss_mask = np.ones(len(sequence) - 1, dtype=np.int32)
        n_pad = self.window_size_plus_one - len(sequence)
        if n_pad > 0:
            sequence = np.pad(sequence, (0, n_pad), 'constant', constant_values=0)
            time = np.pad(time, (0, n_pad), 'constant', constant_values=0)
            loss_mask = np.pad(loss_mask, (0, n_pad), 'constant', constant_values=0)

        sequence = np.array(sequence, dtype=np.float32)
        time = np.array(time, dtype=np.float32)
        return {
            'input_ids': sequence[:-1],
            'labels': sequence[1:],
            'loss_masks': loss_mask,
            'time_values': time[:-1],
        }

    def _find_sequence(self, window_idx):
        """Binary search to find which sequence a window belongs to."""
        lo, hi = 0, len(self.cum_windows) - 2
        while lo <= hi:
            mid = (lo + hi) // 2
            if self.cum_windows[mid] <= window_idx and self.cum_windows[mid + 1] > window_idx:
                return mid
            elif self.cum_windows[mid] > window_idx:
                hi = mid - 1
            else:
                lo = mid + 1
        return lo
