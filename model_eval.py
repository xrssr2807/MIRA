# Copyright (c) Microsoft
# Licensed under MIT

import torch
import torch.nn.functional as F
import json
import argparse
import os
import pickle
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mira.models.modeling_mira import MIRAForPrediction
from mira.models.utils_time_normalization import normalize_time_for_ctrope


def load_pkl_timeseries(pkl_dir, split_path=None):
    """Load PPKL files from a directory. Each pkl contains uid, data(1xT), sampling_rate, label."""
    if split_path and os.path.exists(split_path):
        with open(split_path) as f:
            split = json.load(f)
        file_names = split.get("test", split.get("train", []))
        pkl_files = [os.path.join(pkl_dir, fn) for fn in file_names]
        pkl_files = [fp for fp in pkl_files if os.path.exists(fp)]
    else:
        pkl_files = sorted(glob.glob(os.path.join(pkl_dir, "*.pkl")))

    seqs, times = [], []
    for fp in pkl_files:
        with open(fp, "rb") as f:
            obj = pickle.load(f)
        data = obj["data"]  # shape (1, T)
        sr = obj.get("sampling_rate", 100)
        seq = torch.tensor(data[0], dtype=torch.float32)
        tms = torch.arange(len(seq), dtype=torch.float32) / sr  # time in seconds
        seqs.append(seq)
        times.append(tms)
    return seqs, times


def load_jsonl_timeseries(jsonl_path):
    seqs, times = [], []
    with open(jsonl_path, "r") as f:
        for line in f:
            obj = json.loads(line)
            seqs.append(torch.tensor(obj["sequence"], dtype=torch.float32))
            times.append(torch.tensor(obj["time"], dtype=torch.float32))
    return seqs, times  # keep as lists (no stack!)


def snap_and_dedup_times(t_scaled, snap=0.1):
    snapped = torch.round(t_scaled / snap) * snap
    eps = 1e-4
    for i in range(1, snapped.numel()):
        if snapped[0, i] <= snapped[0, i - 1]:
            snapped[0, i] = snapped[0, i - 1] + eps
    return snapped


def mira_predict_autoreg_norm(model, values, raw_times, C, P, mean, std):
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
                next_target_time_values=None,
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

def evaluate_one_window(model, seq, times, C, P, mean, std):
    """Evaluate only one window (batch size = 1)."""
    device = next(model.parameters()).device

    T = len(seq)
    if T < C + P:
        return None, None, None, None

    # Move sequence and time to device
    hist = seq[:C + P].to(device)
    t_hist = times[:C + P].to(device)

    mean = mean.to(device)
    std = std.to(device)

    pred = mira_predict_autoreg_norm(
        model,
        hist.unsqueeze(0),
        t_hist.unsqueeze(0),
        C,
        P,
        mean,
        std,
    )

    gt = hist[C:C + P].to(device)

    rmse = torch.sqrt(F.mse_loss(pred, gt)).item()
    mae = F.l1_loss(pred, gt).item()
    return rmse, mae, pred.cpu().numpy(), gt.cpu().numpy()


def rolling_eval_dataset(model, seq_list, time_list, settings, out_dir=None):
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    all_predictions = {}

    for C, P in settings:
        rmses, maes = [], []
        viz_preds, viz_gts, viz_ctxs = [], [], []

        for idx, (seq, tms) in enumerate(zip(seq_list, time_list)):

            device = next(model.parameters()).device
            mean = seq.mean().to(device)
            std = (seq.std() + 1e-6).to(device)

            rmse, mae, pred, gt = evaluate_one_window(model, seq, tms, C, P, mean, std)
            if rmse is not None:
                rmses.append(rmse)
                maes.append(mae)
                if out_dir and len(viz_preds) < 5:
                    viz_preds.append(pred)
                    viz_gts.append(gt)
                    viz_ctxs.append(seq[:C].numpy())

        results = {
            "rmse": sum(rmses) / len(rmses) if rmses else float("nan"),
            "mae": sum(maes) / len(maes) if maes else float("nan"),
            "n": len(rmses),
        }

        print(f"{C}->{P} | N={len(rmses)} | RMSE={results['rmse']:.4f} | MAE={results['mae']:.4f}")

        # Save all predictions as numpy
        if out_dir:
            all_predictions[f"{C}_{P}"] = {
                "rmse": results["rmse"],
                "mae": results["mae"],
                "n": results["n"],
                "predictions": viz_preds,
                "ground_truths": viz_gts,
                "contexts": viz_ctxs,
            }

            # Visualization
            if viz_preds:
                fig, axes = plt.subplots(1, min(5, len(viz_preds)), figsize=(20, 3))
                if len(viz_preds) == 1:
                    axes = [axes]
                for ax_i in range(min(5, len(viz_preds))):
                    ax = axes[ax_i]
                    ctx = viz_ctxs[ax_i]
                    gt = viz_gts[ax_i]
                    pr = viz_preds[ax_i]
                    ctx_x = range(max(0, C - 15), C)
                    ctx_y = ctx[max(0, C - 15):]
                    gt_x = range(C, C + len(gt))
                    ax.plot(ctx_x, ctx_y, 'b-', label='Context (last 15)', linewidth=2)
                    ax.plot(gt_x, gt, 'b-', label='Ground Truth', linewidth=2)
                    ax.plot(gt_x, pr, 'r--', label='Prediction', linewidth=2)
                    ax.axvline(x=C, color='gray', linestyle=':', alpha=0.5)
                    ax.set_title(f'Sample {ax_i+1} | RMSE={rmses[ax_i]:.4f}')
                    ax.legend(fontsize=7)
                    ax.grid(True, alpha=0.3)
                fig.suptitle(f'{C}->{P} Prediction Results', fontsize=14)
                fig.tight_layout()
                fig.savefig(f"{out_dir}/viz_{C}_{P}.png", dpi=150)
                plt.close(fig)

    return all_predictions


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--sample", type=int, default=None, help="Limit number of sequences to evaluate")
    parser.add_argument("--out_dir", type=str, default="./eval_results", help="Output directory for predictions and visualizations")
    args = parser.parse_args()

    print("[INFO] Loading model:", args.model)
    model = MIRAForPrediction.from_pretrained(args.model).cuda()
    model.eval()

    print("[INFO] Loading dataset:", args.data)
    data_path = args.data
    if os.path.isdir(data_path):
        split_path = os.path.join(data_path, "split.json")
        seq_list, time_list = load_pkl_timeseries(
            os.path.join(data_path, "data") if os.path.isdir(os.path.join(data_path, "data")) else data_path,
            split_path=split_path if os.path.exists(split_path) else None,
        )
    else:
        seq_list, time_list = load_jsonl_timeseries(data_path)
    print("Loaded:", len(seq_list), "series")

    if args.sample is not None and args.sample < len(seq_list):
        seq_list = seq_list[:args.sample]
        time_list = time_list[:args.sample]
        print("Sampling to", args.sample, "series")

    settings = [
        (48, 24),
        (72, 36),
        (96, 48),
        (128, 64),
    ]

    results = rolling_eval_dataset(model, seq_list, time_list, settings, out_dir=args.out_dir)

    print("\n===== FINAL SUMMARY =====")
    for key, info in results.items():
        print(f"{key}: RMSE={info['rmse']:.4f}, MAE={info['mae']:.4f}, N={info['n']}")

    # Save summary
    os.makedirs(args.out_dir, exist_ok=True)
    summary = {}
    for key, info in results.items():
        summary[key] = {"rmse": info["rmse"], "mae": info["mae"], "n": info["n"]}
    with open(f"{args.out_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[INFO] Summary saved to {args.out_dir}/summary.json")
    print(f"[INFO] Visualizations saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
