#!/usr/bin/env python3
"""Convert PKL files to a single numpy memmap file for fast random access training.
Only extracts the PPG channel (last channel) and ignores ACC channels.
"""

import os
import pickle
import glob
import numpy as np
import gc

PKL_DIR = "/home/zjl/MIRA/ppg_full"
OUTPUT_DIR = "/home/zjl/MIRA/ppg_full"
DATA_FILE = os.path.join(OUTPUT_DIR, "ppg_data.npy")
OFFSETS_FILE = os.path.join(OUTPUT_DIR, "ppg_offsets.npy")

PPG_CHANNEL = 3  # Index of PPG channel (ACC X=0, ACC Y=1, ACC Z=2, PPG=3)


def convert():
    pkl_files = sorted(glob.glob(os.path.join(PKL_DIR, "*.pkl")))
    print(f"Found {len(pkl_files)} pkl files")
    print(f"Extracting PPG channel (index {PPG_CHANNEL}) only, ignoring ACC channels.")

    # First pass: count total elements and sequences
    print("Counting total data points...")
    total_elements = 0
    total_sequences = 0
    seq_lengths = []

    for pkl_path in pkl_files:
        with open(pkl_path, 'rb') as f:
            items = pickle.load(f)
        for item in items:
            seq = item["data"]  # ndarray, shape (n_channels, seq_len)
            if isinstance(seq, np.ndarray) and seq.ndim == 2:
                # Only take PPG channel
                ppg_data = seq[PPG_CHANNEL]
                seq_lengths.append(len(ppg_data))
                total_elements += len(ppg_data)
                total_sequences += 1
            elif isinstance(seq, np.ndarray) and seq.ndim == 1:
                seq_lengths.append(len(seq))
                total_elements += len(seq)
                total_sequences += 1
        gc.collect()

    print(f"Total sequences: {total_sequences}")
    print(f"Total data points: {total_elements}")
    print(f"Expected file size: {total_elements * 4 / 1e9:.1f} GB")

    # Create memmap file
    print(f"Creating memmap file: {DATA_FILE}")
    data = np.memmap(DATA_FILE, dtype='float32', mode='w+', shape=(total_elements,))

    # Build cumulative offsets
    seq_offsets = np.zeros(total_sequences + 1, dtype=np.int64)
    seq_offsets[0] = 0
    for i in range(len(seq_lengths)):
        seq_offsets[i + 1] = seq_offsets[i] + seq_lengths[i]

    print(f"Saving offsets to {OFFSETS_FILE}...")
    np.save(OFFSETS_FILE, seq_offsets)

    # Second pass: fill memmap
    print("Filling memmap...")
    current_offset = 0
    for pkl_path in pkl_files:
        with open(pkl_path, 'rb') as f:
            items = pickle.load(f)
        for item in items:
            seq = item["data"]
            if isinstance(seq, np.ndarray) and seq.ndim == 2:
                ch_data = seq[PPG_CHANNEL].astype(np.float32)
                n = len(ch_data)
                data[current_offset:current_offset + n] = ch_data
                current_offset += n
            elif isinstance(seq, np.ndarray) and seq.ndim == 1:
                ch_data = seq.astype(np.float32)
                n = len(ch_data)
                data[current_offset:current_offset + n] = ch_data
                current_offset += n
        gc.collect()
        print(f"  Written up to offset {current_offset}...")

    del data
    gc.collect()
    print(f"\nDone! Total {current_offset} elements written to {DATA_FILE}")
    print(f"Offsets saved to {OFFSETS_FILE}")
    print(f"File size: {os.path.getsize(DATA_FILE) / 1e9:.1f} GB")


if __name__ == "__main__":
    convert()
