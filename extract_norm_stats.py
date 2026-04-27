#!/usr/bin/env python3
"""
Extract GLOBAL z-score stats (mean, std) from memmap training data.
Save to a single PKL file for downstream tasks (training + evaluation).
"""

import os
import pickle
import numpy as np

DATA_DIR = "/home/zjl/MIRA/ppg_full"
OUTPUT_PATH = os.path.join(DATA_DIR, "norm_stats.pkl")


def extract_stats():
    data_path = os.path.join(DATA_DIR, "ppg_data.npy")
    offsets_path = os.path.join(DATA_DIR, "ppg_offsets.npy")

    if not os.path.exists(data_path) or not os.path.exists(offsets_path):
        print(f"Error: {data_path} or {offsets_path} not found. Run convert_pkl_to_npy.py first.")
        return

    offsets = np.load(offsets_path)
    total_len = int(offsets[-1])
    data = np.memmap(data_path, dtype='float32', mode='r', shape=(total_len,))

    num_sequences = len(offsets) - 1
    print(f"Computing GLOBAL z-score stats from {num_sequences} sequences, {total_len} total points...")

    # Sample for efficiency
    sample_size = min(total_len, 50_000_000)
    if sample_size < total_len:
        rng = np.random.RandomState(42)
        indices = rng.choice(total_len, sample_size, replace=False)
        sampled = data[indices]
        global_mean = float(sampled.mean())
        global_std = float(sampled.std())
    else:
        global_mean = float(data.mean())
        global_std = float(data.std())

    if global_std == 0:
        global_std = 1.0

    print(f"\nGlobal mean: {global_mean:.4f}")
    print(f"Global std:  {global_std:.4f}")

    stats = {"global_mean": global_mean, "global_std": global_std}
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(stats, f)

    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    extract_stats()
