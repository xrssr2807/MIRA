#!/usr/bin/env python3
"""
MIRA 模型性能评估
- 全局 z-score 归一化：训练集计算统一的 mean/std
- 模型在归一化空间训练和推理
- 反归一化参数保存到 PKL 文件，供下游任务使用
"""
import os
import sys
import json
import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from mira.models.modeling_mira import MIRAForPrediction

# Paths
DATA_DIR = "/home/zjl/MIRA/processed_dataset/data"
SPLIT_PATH = "/home/zjl/MIRA/processed_dataset/split.json"
CHECKPOINT = "/home/zjl/MIRA/ppg_output/checkpoint-7000"
OUTPUT_DIR = "/home/zjl/MIRA/ppg_output/eval_results"
TRAIN_DIR = "/home/zjl/MIRA/ppg_full"
NORM_STATS_PATH = os.path.join(TRAIN_DIR, "norm_stats.pkl")

# Eval settings
CONTEXT_LEN = 300
PRED_LEN = 100


def load_norm_stats(path):
    """Load global normalization stats from PKL."""
    with open(path, 'rb') as f:
        stats = pickle.load(f)
    return stats["global_mean"], stats["global_std"]


def load_test_files(split_path, data_dir, max_samples=500):
    """Load test PKL files. Only PPG channel."""
    with open(split_path) as f:
        split = json.load(f)
    test_files = split.get("test", [])
    print(f"Total test files: {len(test_files)}")

    seqs = []
    for fn in test_files[:max_samples]:
        fp = os.path.join(data_dir, fn)
        if not os.path.exists(fp):
            continue
        with open(fp, 'rb') as f:
            item = pickle.load(f)
        data = item["data"]
        if data.ndim == 2:
            data = data[0]
        data = data.astype(np.float32)
        if len(data) >= CONTEXT_LEN + PRED_LEN:
            seqs.append(data)

    print(f"Loaded {len(seqs)} sequences with length >= {CONTEXT_LEN + PRED_LEN}")
    return seqs


def normalize_sequence(seq, global_mean, global_std):
    """Z-score normalize using GLOBAL mean/std."""
    return (seq - global_mean) / global_std


def predict_normalized(model, context_norm, pred_len, device):
    """Autoregressive prediction in normalized space."""
    model.eval()
    cur_input = torch.tensor(context_norm, dtype=torch.float32).unsqueeze(-1).to(device)
    predictions = []

    for _ in range(pred_len):
        with torch.no_grad():
            seq_len = cur_input.shape[0]
            input_ids = cur_input.unsqueeze(0)
            time_values = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(0) * 10.0
            out = model(input_ids=input_ids, time_values=time_values, return_dict=True)
            next_pred = out.logits[0, -1]
            predictions.append(next_pred.cpu().item())
            cur_input = torch.cat([cur_input, next_pred.unsqueeze(-1)], dim=0)

    return np.array(predictions)


def inverse_normalize(preds_norm, global_mean, global_std):
    return preds_norm * global_std + global_mean


def evaluate(seqs, model, device, global_mean, global_std, max_samples=200):
    all_rmse, all_mae, all_mape = [], [], []
    viz_pairs = []

    total = min(len(seqs), max_samples)
    print(f"\nEvaluating {total} sequences...")
    print(f"Global mean={global_mean:.4f}, std={global_std:.4f}")

    for i in range(total):
        seq = seqs[i]
        ground_truth = seq[CONTEXT_LEN:CONTEXT_LEN + PRED_LEN]

        # Normalize with GLOBAL stats
        seq_norm = normalize_sequence(seq, global_mean, global_std)
        context_norm = seq_norm[:CONTEXT_LEN]

        # Predict in normalized space
        preds_norm = predict_normalized(model, context_norm, PRED_LEN, device)

        # Inverse normalize with GLOBAL stats
        preds = inverse_normalize(preds_norm, global_mean, global_std)

        rmse = np.sqrt(np.mean((preds - ground_truth) ** 2))
        mae = np.mean(np.abs(preds - ground_truth))
        mape = np.mean(np.abs((ground_truth - preds) / (np.abs(ground_truth) + 1e-8))) * 100

        all_rmse.append(rmse)
        all_mae.append(mae)
        all_mape.append(mape)

        if i < 5:
            viz_pairs.append((seq[:CONTEXT_LEN], ground_truth, preds, rmse))

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{total}, avg RMSE={np.mean(all_rmse):.4f}")

    return {
        "rmse": float(np.mean(all_rmse)),
        "mae": float(np.mean(all_mae)),
        "mape": float(np.mean(all_mape)),
        "n": len(all_rmse),
    }, viz_pairs


def plot_results(viz_pairs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(len(viz_pairs), 1, figsize=(14, 3 * len(viz_pairs)))
    if len(viz_pairs) == 1:
        axes = [axes]

    for i, (ctx, gt, pred, rmse) in enumerate(viz_pairs):
        ax = axes[i]
        total_len = CONTEXT_LEN + PRED_LEN
        x = np.arange(total_len)
        ax.plot(x[:CONTEXT_LEN], ctx, 'b-', linewidth=1.5, label='Context (300)', alpha=0.7)
        ax.plot(x[CONTEXT_LEN - 10:CONTEXT_LEN + PRED_LEN],
                np.concatenate([ctx[-10:], gt]), 'g-', linewidth=1.5, label='Ground Truth', alpha=0.8)
        ax.plot(x[CONTEXT_LEN:CONTEXT_LEN + PRED_LEN], pred, 'r--', linewidth=1.5,
                label=f'Prediction (RMSE={rmse:.4f})')
        ax.axvline(x=CONTEXT_LEN, color='gray', linestyle=':', alpha=0.5)
        ax.set_title(f'Sample {i + 1}')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'MIRA Evaluation: {CONTEXT_LEN} context -> {PRED_LEN} prediction',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    plot_path = os.path.join(output_dir, 'predictions.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved visualization to {plot_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Context: {CONTEXT_LEN} points -> Predict: {PRED_LEN} points\n")

    # Load global normalization stats
    print(f"Loading global norm stats from {NORM_STATS_PATH}...")
    global_mean, global_std = load_norm_stats(NORM_STATS_PATH)
    print(f"  Global mean: {global_mean:.4f}")
    print(f"  Global std:  {global_std:.4f}\n")

    print("Loading model...")
    model = MIRAForPrediction.from_pretrained(
        CHECKPOINT, local_files_only=True, attn_implementation='eager',
    ).to(device)
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M\n")

    print("Loading test data...")
    seqs = load_test_files(SPLIT_PATH, DATA_DIR, max_samples=1000)
    print()

    metrics, viz_pairs = evaluate(seqs, model, device, global_mean, global_std, max_samples=200)

    print("\n" + "=" * 50)
    print("  EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Samples evaluated: {metrics['n']}")
    print(f"  Context -> Predict: {CONTEXT_LEN} -> {PRED_LEN}")
    print(f"  Global mean: {global_mean:.4f}, std: {global_std:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print("=" * 50)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = {
        "checkpoint": CHECKPOINT,
        "context_len": CONTEXT_LEN,
        "pred_len": PRED_LEN,
        "global_mean": global_mean,
        "global_std": global_std,
        "n_samples": metrics["n"],
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "mape": metrics["mape"],
    }
    with open(os.path.join(OUTPUT_DIR, "eval_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to {OUTPUT_DIR}/eval_results.json")

    # Save global norm stats (PKL) for downstream tasks
    with open(os.path.join(OUTPUT_DIR, "norm_stats.pkl"), 'wb') as f:
        pickle.dump({"global_mean": global_mean, "global_std": global_std}, f)
    print(f"Saved global norm stats to {OUTPUT_DIR}/norm_stats.pkl")

    if viz_pairs:
        plot_results(viz_pairs, OUTPUT_DIR)


if __name__ == "__main__":
    main()
