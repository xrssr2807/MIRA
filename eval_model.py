#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MIRA Model Evaluation Script
- Autoregressive prediction on multiple windows
- Computes RMSE, MAE at different context/prediction lengths
- Visualizes prediction results
"""
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mira.models.modeling_mira import MIRAForPrediction
from mira.models.utils_time_normalization import normalize_time_for_ctrope


def load_jsonl_timeseries(jsonl_path):
    seqs, times = [], []
    with open(jsonl_path, "r") as f:
        for line in f:
            obj = json.loads(line)
            seqs.append(torch.tensor(obj["sequence"], dtype=torch.float32))
            times.append(torch.tensor(obj["time"], dtype=torch.float32))
    return seqs, times


def snap_and_dedup_times(t_scaled, snap=0.1):
    snapped = torch.round(t_scaled / snap) * snap
    eps = 1e-4
    for i in range(1, snapped.numel()):
        if snapped[0, i] <= snapped[0, i - 1]:
            snapped[0, i] = snapped[0, i - 1] + eps
    return snapped


def mira_predict_autoreg(model, values, raw_times, C, P, mean, std):
    device = next(model.parameters()).device
    values = values.to(device)
    raw_times = raw_times.to(device)
    mean = mean.to(device)
    std = std.to(device)

    values_norm = (values - mean) / std

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
            out = model(
                input_ids=inp_vals,
                time_values=inp_times,
                return_dict=True,
            )
        next_norm = out.logits[:, -1, :]
        preds_norm.append(next_norm.squeeze(0))

        next_t = future_times[:, i:i + 1]
        cur_vals = torch.cat([cur_vals, next_norm], dim=1)
        cur_times = torch.cat([cur_times, next_t], dim=1)

    preds_norm = torch.stack(preds_norm, dim=1)
    preds = preds_norm * std + mean
    return preds.squeeze(0)


def evaluate_one_window(model, seq, times, C, P):
    T = len(seq)
    if T < C + P:
        return None, None, None, None

    device = next(model.parameters()).device
    mean = seq.mean()
    std = seq.std() + 1e-6

    hist = seq[:C + P]
    t_hist = times[:C + P]

    pred = mira_predict_autoreg(
        model, hist.unsqueeze(0), t_hist.unsqueeze(0), C, P, mean, std
    )
    gt = hist[C:C + P].to(device)

    rmse = torch.sqrt(F.mse_loss(pred, gt)).item()
    mae = F.l1_loss(pred, gt).item()
    return pred.cpu(), gt.cpu(), rmse, mae


def rolling_eval(model, seq_list, time_list, settings, sample=200, viz_dir=None):
    results = {}

    for C, P in settings:
        rmses, maes = [], []
        viz_count = 0
        viz_preds, viz_gts = [], []

        # Evaluate up to `sample` sequences
        total = min(len(seq_list), sample)
        for idx in range(total):
            seq, tms = seq_list[idx], time_list[idx]
            pred, gt, rmse, mae = evaluate_one_window(model, seq, tms, C, P)
            if rmse is not None:
                rmses.append(rmse)
                maes.append(mae)
                if viz_count < 5 and viz_dir:
                    viz_preds.append(pred.numpy())
                    viz_gts.append(gt.numpy())
                    viz_count += 1

        avg_rmse = np.mean(rmses) if rmses else float("nan")
        avg_mae = np.mean(maes) if maes else float("nan")

        results[(C, P)] = {
            "rmse": avg_rmse,
            "mae": avg_mae,
            "n": len(rmses),
        }
        print(f"  {C}->{P} | N={len(rmses)} | RMSE={avg_rmse:.4f} | MAE={avg_mae:.4f}")

        # Visualization
        if viz_dir and viz_preds:
            fig, axes = plt.subplots(1, min(5, len(viz_preds)), figsize=(20, 3))
            if len(viz_preds) == 1:
                axes = [axes]
            for ax_i in range(min(5, len(viz_preds))):
                ax = axes[ax_i]
                ctx_vals = seq_list[idx].numpy()[:C] if idx < len(seq_list) else viz_gts[ax_i]
                ax.plot(range(C - 10, C), ctx_vals[-10:], 'b-', label='Context (last 10)', linewidth=2)
                ax.plot(range(C, C + P), viz_gts[ax_i], 'b-', label='Ground Truth', linewidth=2)
                ax.plot(range(C, C + P), viz_preds[ax_i], 'r--', label='Prediction', linewidth=2)
                ax.set_title(f'Sample {ax_i+1} | RMSE={rmses[ax_i]:.4f}')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            fig.suptitle(f'{C}->{P} Prediction Results', fontsize=14)
            fig.tight_layout()
            fig.savefig(f"{viz_dir}/viz_{C}_{P}.png", dpi=150)
            plt.close(fig)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, required=True, help="Model path")
    parser.add_argument("--data", "-d", type=str, required=True, help="JSONL data path")
    parser.add_argument("--sample", "-s", type=int, default=200, help="Number of samples to evaluate")
    parser.add_argument("--viz_dir", type=str, default="./eval_viz", help="Visualization output dir")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading model from {args.model}...")
    model = MIRAForPrediction.from_pretrained(args.model).to(device)
    model.eval()
    print(f"[INFO] Model loaded. Device: {device}")

    print(f"[INFO] Loading data from {args.data}...")
    seq_list, time_list = load_jsonl_timeseries(args.data)
    print(f"[INFO] Loaded {len(seq_list)} sequences")

    settings = [
        (48, 24),    # short: 480ms context -> 240ms pred
        (96, 48),    # medium: 960ms -> 480ms
        (128, 64),   # longer: 1.28s -> 640ms
        (256, 128),  # long: 2.56s -> 1.28s
    ]

    import os
    os.makedirs(args.viz_dir, exist_ok=True)

    print("\n===== Running Evaluation =====")
    results = rolling_eval(model, seq_list, time_list, settings, sample=args.sample, viz_dir=args.viz_dir)

    print("\n===== FINAL SUMMARY =====")
    for (C, P), info in results.items():
        print(f"  {C}->{P}: RMSE={info['rmse']:.4f}, MAE={info['mae']:.4f}, N={info['n']}")

    # Save results
    import json
    summary = {}
    for (C, P), info in results.items():
        summary[f"{C}->{P}"] = info
    with open(f"{args.viz_dir}/results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[INFO] Results saved to {args.viz_dir}/results.json")
    print(f"[INFO] Visualizations saved to {args.viz_dir}/")


if __name__ == "__main__":
    main()
