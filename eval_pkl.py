#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MIRA Model Evaluation Script for PKL data.
Runs on remote server, evaluates model on .pkl time-series data.
"""
import os
import sys
import pickle
import argparse
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Add MIRA to path so custom model classes are registered with transformers
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from MIRA.models.modeling_mira import MIRAForPrediction
from MIRA.models.utils_time_normalization import normalize_time_for_ctrope


def load_pkl_sequences(data_path, sample_size=None):
    """Load sequences from .pkl files with flexible structure detection."""
    if os.path.isfile(data_path) and data_path.endswith('.pkl'):
        pkl_files = [data_path]
    elif os.path.isdir(data_path):
        pkl_files = sorted(
            [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.pkl')]
        )
    else:
        raise ValueError(f"Not a .pkl file or directory: {data_path}")

    print(f"  Found {len(pkl_files)} .pkl files")

    seqs, times = [], []
    total_files = 0
    skipped_items = 0

    # Peek first file to detect structure
    with open(pkl_files[0], 'rb') as f:
        first_item = pickle.load(f)

    # Determine format
    if isinstance(first_item, list) and len(first_item) > 0 and isinstance(first_item[0], dict):
        fmt = "list_of_dicts"  # [{"data": ...}, {"data": ...}]
    elif isinstance(first_item, dict) and "data" in first_item:
        fmt = "single_dict"     # {"uid": ..., "data": ndarray, "label": ...}
    elif isinstance(first_item, list) and len(first_item) > 0:
        fmt = "list"            # [ndarray, ndarray, ...]
    else:
        fmt = "direct"          # ndarray directly

    print(f"  Detected format: {fmt} (first file: {os.path.basename(pkl_files[0])})")

    for pkl_path in pkl_files:
        total_files += 1
        try:
            with open(pkl_path, 'rb') as f:
                items = pickle.load(f)

            if fmt == "list_of_dicts":
                for item in items:
                    if isinstance(item, dict) and "data" in item:
                        data = item["data"]
                    else:
                        data = item
                    data = np.array(data, dtype=np.float32)
                    if data.ndim == 2:
                        for ch in range(data.shape[0]):
                            seqs.append(data[ch].astype(np.float32))
                            times.append(np.arange(len(data[ch]), dtype=np.float32) * 10.0)
                    elif data.ndim == 1:
                        seqs.append(data.astype(np.float32))
                        times.append(np.arange(len(data), dtype=np.float32) * 10.0)
                    else:
                        skipped_items += 1

            elif fmt == "single_dict":
                data = items.get("data", items)
                data = np.array(data, dtype=np.float32)
                if data.ndim == 2:
                    for ch in range(data.shape[0]):
                        seqs.append(data[ch].astype(np.float32))
                        times.append(np.arange(len(data[ch]), dtype=np.float32) * 10.0)
                elif data.ndim == 1:
                    seqs.append(data.astype(np.float32))
                    times.append(np.arange(len(data), dtype=np.float32) * 10.0)
                else:
                    skipped_items += 1

            elif fmt == "list":
                for item in items:
                    data = np.array(item, dtype=np.float32)
                    if data.ndim == 2:
                        for ch in range(data.shape[0]):
                            seqs.append(data[ch].astype(np.float32))
                            times.append(np.arange(len(data[ch]), dtype=np.float32) * 10.0)
                    elif data.ndim == 1:
                        seqs.append(data.astype(np.float32))
                        times.append(np.arange(len(data), dtype=np.float32) * 10.0)
                    else:
                        skipped_items += 1

            elif fmt == "direct":
                data = np.array(items, dtype=np.float32)
                if data.ndim == 2:
                    for ch in range(data.shape[0]):
                        seqs.append(data[ch].astype(np.float32))
                        times.append(np.arange(len(data[ch]), dtype=np.float32) * 10.0)
                elif data.ndim == 1:
                    seqs.append(data.astype(np.float32))
                    times.append(np.arange(len(data), dtype=np.float32) * 10.0)
                else:
                    skipped_items += 1

        except Exception as e:
            if total_files <= 5:
                print(f"  Warning: failed {os.path.basename(pkl_path)}: {e}")

    if not seqs:
        raise ValueError("No sequences loaded. Check .pkl file structure.")

    print(f"  Total: {len(seqs)} sequences, {len(seqs[0])} points each")

    if sample_size and sample_size < len(seqs):
        rng = np.random.RandomState(42)
        indices = rng.choice(len(seqs), sample_size, replace=False)
        indices = np.sort(indices)
        seqs = [seqs[i] for i in indices]
        times = [times[i] for i in indices]
        print(f"  Sampled {len(seqs)} sequences (seed=42)")

    return seqs, times


def snap_and_dedup_times(t, snap=0.1):
    """Snap and deduplicate times for CT-RoPE. Supports [B, seq_len]."""
    snapped = torch.round(t / snap) * snap
    eps = 1e-4
    for b in range(snapped.shape[0]):
        for i in range(1, snapped.shape[1]):
            if snapped[b, i] <= snapped[b, i - 1]:
                snapped[b, i] = snapped[b, i - 1] + eps
    return snapped


def mira_predict_autoreg(model, values, raw_times, C, P):
    """Autoregressive prediction: context C -> predict P steps. Supports batched input."""
    device = next(model.parameters()).device
    values = values.to(device)
    raw_times = raw_times.to(device)

    # Normalize
    mean = values.mean(dim=1, keepdim=True)  # [B, 1]
    std = values.std(dim=1, keepdim=True) + 1e-6  # [B, 1]
    values_norm = (values - mean) / std

    # Time normalization for CT-RoPE (matches training)
    full_scaled_times, t_min, t_max = normalize_time_for_ctrope(
        time_values=raw_times,
        attention_mask=torch.ones_like(raw_times),
        seq_length=raw_times.shape[1],
        alpha=1.0,
    )
    full_scaled_times = snap_and_dedup_times(full_scaled_times)

    hist_vals = values_norm[:, :C]
    hist_times = full_scaled_times[:, :C]
    future_times = full_scaled_times[:, C:C + P]

    cur_vals = hist_vals.clone()
    cur_times = hist_times.clone()

    preds_norm = []
    for i in range(P):
        inp_vals = cur_vals.unsqueeze(-1)
        inp_times = cur_times

        with torch.no_grad():
            with torch.autocast("cuda", enabled=torch.cuda.is_available()):
                out = model(
                    input_ids=inp_vals,
                    time_values=inp_times,
                    return_dict=True,
                )
        next_norm = out.logits[:, -1, :]
        preds_norm.append(next_norm)

        next_t = future_times[:, i:i + 1]
        cur_vals = torch.cat([cur_vals, next_norm], dim=1)
        cur_times = torch.cat([cur_times, next_t], dim=1)

    preds_norm = torch.stack(preds_norm, dim=1)  # [B, P, D]
    preds = preds_norm * std.unsqueeze(1) + mean.unsqueeze(1)
    return preds, mean, std


def evaluate_batch(model, seq_list, time_list, C, P, device):
    """Evaluate a batch of sequences at once. Returns list of (pred, gt, rmse, mae)."""
    batch_size = len(seq_list)

    # Pad sequences to same length (C+P)
    target_len = C + P
    hist_list, t_hist_list = [], []
    valid_mask = []

    for seq, tms in zip(seq_list, time_list):
        if len(seq) < target_len:
            valid_mask.append(False)
            continue
        valid_mask.append(True)
        hist_list.append(torch.from_numpy(seq[:target_len]).float())
        t_hist_list.append(torch.from_numpy(tms[:target_len]).float())

    if not hist_list:
        return []

    hist = torch.stack(hist_list).to(device)        # [B, C+P]
    t_hist = torch.stack(t_hist_list).to(device)    # [B, C+P]

    # Autoregressive prediction
    pred, mean, std = mira_predict_autoreg(
        model, hist, t_hist, C, P
    )  # [B, P, D]

    gt = hist[:, C:C + P]  # [B, P, D]

    # Compute per-sample metrics
    results = []
    for i in range(batch_size):
        if not valid_mask[i]:
            results.append(None)
            continue
        p = pred[i].squeeze(-1)
        g = gt[i].squeeze(-1)
        rmse = torch.sqrt(F.mse_loss(p, g)).item()
        mae = F.l1_loss(p, g).item()
        results.append((p.cpu(), g.cpu(), rmse, mae))

    return results


def rolling_eval(model, seq_list, time_list, settings, device, batch_size=64, viz_dir=None):
    """Rolling evaluation across settings, with batched inference."""
    results = {}
    total_seqs = len(seq_list)

    for C, P in settings:
        rmses, maes = [], []
        viz_count = 0
        viz_preds, viz_gts, viz_contexts = [], [], []
        skipped = 0

        # Process in batches
        n_batches = (total_seqs + batch_size - 1) // batch_size
        iterator = range(n_batches)
        if HAS_TQDM:
            iterator = tqdm(iterator, total=n_batches, desc=f"  {C}->{P}")

        for batch_idx in iterator:
            start = batch_idx * batch_size
            end = min(start + batch_size, total_seqs)
            batch_seqs = seq_list[start:end]
            batch_times = time_list[start:end]

            batch_results = evaluate_batch(model, batch_seqs, batch_times, C, P, device)

            for i, res in enumerate(batch_results):
                if res is None:
                    skipped += 1
                    continue
                pred, gt, rmse, mae = res
                rmses.append(rmse)
                maes.append(mae)

                if viz_count < 5 and viz_dir:
                    viz_preds.append(pred.numpy())
                    viz_gts.append(gt.numpy())
                    viz_contexts.append(batch_seqs[i][:C])
                    viz_count += 1

        avg_rmse = np.mean(rmses) if rmses else float("nan")
        avg_mae = np.mean(maes) if maes else float("nan")

        results[(C, P)] = {
            "rmse": round(avg_rmse, 6),
            "mae": round(avg_mae, 6),
            "n": len(rmses),
            "skipped": skipped,
            "all_rmse": [round(r, 6) for r in rmses],
            "all_mae": [round(m, 6) for m in maes],
        }
        if HAS_TQDM:
            tqdm.write(
                f"  {C}->{P:3d} | N={len(rmses):5d}/{total_seqs} "
                f"(skip={skipped}) | RMSE={avg_rmse:.4f} | MAE={avg_mae:.4f}"
            )
        else:
            print(
                f"  {C}->{P:3d} | N={len(rmses):5d}/{total_seqs} "
                f"(skip={skipped}) | RMSE={avg_rmse:.4f} | MAE={avg_mae:.4f}"
            )

        # Visualization
        if viz_dir and viz_preds:
            fig, axes = plt.subplots(1, len(viz_preds), figsize=(4 * len(viz_preds), 3))
            if len(viz_preds) == 1:
                axes = [axes]
            for ax_i, ax in enumerate(axes):
                ctx = viz_contexts[ax_i]
                gt = viz_gts[ax_i]
                pred = viz_preds[ax_i]

                # Context: last 20 points + first point of GT to ensure visual continuity
                ctx_len = min(20, C)
                ctx_x = list(range(C - ctx_len, C + 1))
                ctx_y = ctx[-ctx_len:].tolist() + [gt[0]]

                ax.plot(ctx_x, ctx_y, 'b-', linewidth=2, label='Context')
                ax.plot(range(C, C + P), gt, 'b--', linewidth=2, label='Ground Truth')
                ax.plot(range(C, C + P), pred, 'r-', linewidth=2, label='Prediction')
                ax.set_title(f'Sample {ax_i+1} | RMSE={rmses[ax_i]:.4f}')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

            fig.suptitle(f'{C}->{P} Prediction Results', fontsize=14)
            fig.tight_layout()
            fig.savefig(f"{viz_dir}/viz_{C}_{P}.png", dpi=150)
            plt.close(fig)

    return results


def main():
    parser = argparse.ArgumentParser(description="MIRA Evaluation on PKL data")
    parser.add_argument("--model", "-m", type=str, required=True, help="Path to trained model dir")
    parser.add_argument("--data", "-d", type=str, required=True, help="Path to .pkl file or directory")
    parser.add_argument("--sample", "-s", type=int, default=2000,
                        help="Max sequences to evaluate (default 2000, set --all for full 46827)")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all 46827 sequences (slow)")
    parser.add_argument("--batch_size", "-b", type=int, default=64,
                        help="Batch size for inference (default 64)")
    parser.add_argument("--viz_dir", type=str, default="./eval_viz", help="Visualization output dir")
    args = parser.parse_args()

    sample_size = None if args.all else args.sample

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    # --- Load model ---
    print(f"[INFO] Loading model from {args.model}...")
    model_config_path = os.path.join(args.model, "config.json")
    with open(model_config_path) as f:
        config_dict = json.load(f)

    # Remove auto_map so transformers uses registered model_type="mira"
    config_dict.pop("auto_map", None)
    tmp_config_path = os.path.join(args.model, "_eval_config.json")
    with open(tmp_config_path, "w") as f:
        json.dump(config_dict, f)

    model = MIRAForPrediction.from_pretrained(args.model).to(device)
    model.eval()

    os.remove(tmp_config_path)
    print(f"[INFO] Model loaded successfully.")

    # --- Load data ---
    print(f"[INFO] Loading data from {args.data}...")
    t0 = time.time()
    seq_list, time_list = load_pkl_sequences(args.data, sample_size=sample_size)
    print(f"[INFO] Data loading took {time.time()-t0:.1f}s")
    print(f"[INFO] Evaluating {len(seq_list)} sequences")

    # --- Evaluation settings ---
    # Adjusted for data with ~1000 points per seq
    settings = [
        (48, 24),     # 480ms context  -> 240ms  pred (short)
        (96, 48),     # 960ms  context -> 480ms  pred (medium)
        (128, 64),    # 1.28s  context -> 640ms  pred (longer)
        (256, 128),   # 2.56s  context -> 1.28s  pred (long)
        (512, 256),   # 5.12s  context -> 2.56s  pred (very long)
        (768, 192),   # 7.68s  context -> 1.92s  pred (extended, max for 1000pt seqs)
    ]

    os.makedirs(args.viz_dir, exist_ok=True)

    print("\n===== Running Evaluation =====")
    t0 = time.time()
    results = rolling_eval(model, seq_list, time_list, settings, device, batch_size=args.batch_size, viz_dir=args.viz_dir)
    print(f"\n[INFO] Evaluation took {time.time()-t0:.1f}s")

    print("\n===== FINAL SUMMARY =====")
    for (C, P), info in results.items():
        skip_str = f" (skipped {info['skipped']})" if info.get('skipped', 0) > 0 else ""
        print(f"  {C:4d}->{P:4d}: RMSE={info['rmse']:.4f}, MAE={info['mae']:.4f}, N={info['n']}{skip_str}")

    # Save results
    summary = {}
    for (C, P), info in results.items():
        summary[f"{C}->{P}"] = {
            "rmse": info["rmse"],
            "mae": info["mae"],
            "n": info["n"],
            "skipped": info.get("skipped", 0),
        }
    with open(f"{args.viz_dir}/results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[INFO] Results saved to {args.viz_dir}/results.json")
    print(f"[INFO] Visualizations saved to {args.viz_dir}/")


if __name__ == "__main__":
    main()
