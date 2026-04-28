#!/usr/bin/env python3
"""Convert PKL files to a single numpy memmap file for fast random access training.
Only extracts the PPG channel (last channel) and ignores ACC channels.
"""

import os
import pickle
import glob
import numpy as np
import gc

PKL_DIR = os.environ.get("MIRA_DATA_DIR", "ppg_full")
OUTPUT_DIR = os.environ.get("MIRA_DATA_DIR", "ppg_full")
DATA_FILE = os.path.join(OUTPUT_DIR, "ppg_data.npy")
OFFSETS_FILE = os.path.join(OUTPUT_DIR, "ppg_offsets.npy")



def convert():
    pkl_files = sorted(glob.glob(os.path.join(PKL_DIR, "*.pkl")))
    print(f"Found {len(pkl_files)} pkl files")

    # First pass: count total elements and sequences
    print("Counting total data points...")
    total_elements = 0
    total_sequences = 0
    seq_lengths = []

    for pkl_path in pkl_files:
        with open(pkl_path, 'rb') as f:
            container = pickle.load(f)

        # Support both formats:
        # 1. List of dicts: [{'data': ndarray}, ...]  (old commom1_part_*.pkl, 4 channels, PPG=3)
        # 2. Direct dict: {'timestamp': str, 'data': ndarray}  (new combined_*.pkl, 5 channels, PPG=4)
        if isinstance(container, dict) and 'data' in container:
            seq = container['data']
            ch = 4 if seq.shape[0] >= 5 else seq.shape[0] - 1  # PPG is last channel
            ppg = seq[ch]
            seq_lengths.append(len(ppg))
            total_elements += len(ppg)
            total_sequences += 1
        elif isinstance(container, list):
            for item in container:
                seq = item["data"]
                if isinstance(seq, np.ndarray) and seq.ndim == 2:
                    ch = seq.shape[0] - 1  # PPG is last channel
                    seq_lengths.append(len(seq[ch]))
                    total_elements += len(seq[ch])
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
            container = pickle.load(f)

        if isinstance(container, dict) and 'data' in container:
            seq = container['data']
            ch = 4 if seq.shape[0] >= 5 else seq.shape[0] - 1  # PPG is last channel
            ch_data = seq[ch].astype(np.float32)
            n = len(ch_data)
            data[current_offset:current_offset + n] = ch_data
            current_offset += n
        elif isinstance(container, list):
            for item in container:
                seq = item["data"]
                if isinstance(seq, np.ndarray) and seq.ndim == 2:
                    ch = seq.shape[0] - 1  # PPG is last channel
                    ch_data = seq[ch].astype(np.float32)
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
