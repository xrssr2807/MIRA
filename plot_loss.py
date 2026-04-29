#!/usr/bin/env python3
"""
MIRA 训练 Loss 图生成器
支持两种模式:
  1. 从 2.txt (终端训练日志) 解析
  2. 从 checkpoint/trainer_state.json 读取 (原有模式, 备用)
"""
import os
import sys
import re
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.environ.get("MIRA_OUTPUT_DIR", "ppg_output")
LOG_FILE = os.environ.get("MIRA_LOG_FILE", "2.txt")
INTERVAL = 500


def parse_log_file(log_path):
    """Parse training log from terminal output (2.txt format)"""
    steps, losses, lrs, grads = [], [], [], []
    step = 0
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if "'loss'" not in line or "'train_runtime'" in line:
                continue
            try:
                d = eval(line)
                steps.append(step)
                losses.append(d['loss'])
                lrs.append(d.get('learning_rate', 0))
                grads.append(d.get('grad_norm', 0))
                step += 1
            except Exception:
                continue
    return steps, losses, lrs, grads


def get_trainer_state():
    """Read latest trainer_state.json from checkpoint directory"""
    checkpoints = sorted([
        d for d in os.listdir(OUTPUT_DIR)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(OUTPUT_DIR, d))
    ], key=lambda x: int(x.split("-")[-1]))

    if not checkpoints:
        return None

    state_path = os.path.join(OUTPUT_DIR, checkpoints[-1], "trainer_state.json")
    if not os.path.exists(state_path):
        return None

    with open(state_path) as f:
        return json.load(f)


def extract_data(state):
    """Extract steps, losses, lrs, grads from trainer_state"""
    log_history = state.get("log_history", [])
    steps, losses, lrs, grads = [], [], [], []
    for entry in log_history:
        if "step" not in entry:
            continue
        steps.append(entry["step"])
        losses.append(entry.get("loss", 0))
        lrs.append(entry.get("learning_rate", 0))
        grads.append(entry.get("grad_norm", 0))
    return steps, losses, lrs, grads


def smooth(y, window=20):
    """Moving average smoothing"""
    if len(y) <= window:
        return list(y)
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode='valid').tolist()


def generate_plot_from_log(steps, losses, lrs, grads, max_steps=None):
    """Generate loss plot from parsed log data"""
    if len(steps) < 2:
        return False

    n = len(steps)
    last_step = steps[-1]
    max_steps = max_steps or last_step

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'MIRA Training Loss - Step {last_step} / {max_steps}', fontsize=14, fontweight='bold')

    # Plot 1: Full loss with smoothing
    ax = axes[0, 0]
    ax.plot(steps, losses, 'b-', alpha=0.5, linewidth=0.5, label='Raw loss')
    if n > 50:
        s_loss = smooth(losses, min(50, n // 5))
        offset = len(losses) - len(s_loss)
        ax.plot(steps[offset:], s_loss, 'r-', linewidth=2, label=f'Smooth (w={min(50, n // 5)})')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title(f'Loss (n={n} pts)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Plot 2: Zoom on last portion
    ax = axes[0, 1]
    zoom_count = min(500, max(10, n // 2))
    r_steps = steps[-zoom_count:]
    r_losses = losses[-zoom_count:]
    ax.plot(r_steps, r_losses, 'b-', alpha=0.6, linewidth=0.8)
    if len(r_losses) > 20:
        s_r = smooth(r_losses, min(20, len(r_losses) // 3))
        ax.plot(r_steps[-len(s_r):], s_r, 'r-', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title(f'Last {zoom_count} steps (mean={np.mean(r_losses):.4f}, std={np.std(r_losses):.4f})')
    ax.grid(True, alpha=0.3)

    # Plot 3: Gradient norm (log scale)
    ax = axes[1, 0]
    ax.plot(steps, grads, 'b-', alpha=0.5, linewidth=0.5)
    if n > 50:
        s_grad = smooth(grads, min(50, n // 5))
        offset = len(grads) - len(s_grad)
        ax.plot(steps[offset:], s_grad, 'r-', linewidth=1.5)
    ax.set_yscale('log')
    ax.set_xlabel('Step')
    ax.set_ylabel('Grad Norm (log)')
    ax.set_title(f'Grad Norm (latest={grads[-1]:.1f})')
    ax.grid(True, alpha=0.3)

    # Plot 4: LR schedule + Stats
    ax = axes[1, 1]
    ax2 = ax.twinx()
    ax.plot(steps, lrs, 'g-', linewidth=1.5, label='Learning Rate')
    ax.set_xlabel('Step')
    ax.set_ylabel('Learning Rate', color='g')
    ax.tick_params(axis='y', labelcolor='g')
    pct = last_step / max_steps * 100
    ax.text(0.02, 0.95, f'Progress: {pct:.1f}% ({last_step}/{max_steps})',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    ax.grid(True, alpha=0.3)

    ax2.set_ylabel('Loss Stats', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    stats_text = (
        f"  Latest loss:  {losses[-1]:.4f}\n"
        f"  Min loss:     {min(losses):.4f}\n"
        f"  Mean loss:    {np.mean(losses):.4f}\n"
        f"  Latest LR:    {lrs[-1]:.2e}\n"
        f"  Latest grad:  {grads[-1]:.1f}\n"
    )
    ax2.text(0.02, 0.02, stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='bottom', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f'loss_step{last_step}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    return True


def generate_plot_from_state(state, step_number):
    """Generate loss plot from trainer_state data"""
    steps, losses, lrs, grads = extract_data(state)
    if len(steps) < 2:
        return False

    max_steps = state.get("max_steps", "?")
    n = len(steps)
    last_step = steps[-1]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'MIRA Training Loss - Step {last_step} / {max_steps}', fontsize=14, fontweight='bold')

    # Plot 1: Full loss with smoothing
    ax = axes[0, 0]
    ax.plot(steps, losses, 'b-', alpha=0.5, linewidth=0.5, label='Raw loss')
    if n > 50:
        s_loss = smooth(losses, min(50, n // 5))
        offset = len(losses) - len(s_loss)
        ax.plot(steps[offset:], s_loss, 'r-', linewidth=2, label=f'Smooth (w={min(50, n // 5)})')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title(f'Loss (n={n} pts)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Plot 2: Zoom on last 500 steps
    ax = axes[0, 1]
    recent_mask = np.array(steps) > (last_step - 500)
    if recent_mask.sum() > 1:
        r_steps = np.array(steps)[recent_mask]
        r_losses = np.array(losses)[recent_mask]
        ax.plot(r_steps, r_losses, 'b-', alpha=0.6, linewidth=0.8)
        if len(r_losses) > 20:
            s_r = smooth(r_losses, min(20, len(r_losses) // 3))
            ax.plot(r_steps[-len(s_r):], s_r, 'r-', linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title(f'Last 500 steps (mean={r_losses.mean():.4f}, std={r_losses.std():.4f})')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Not enough data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Last 500 steps')

    # Plot 3: Gradient norm (log scale)
    ax = axes[1, 0]
    ax.plot(steps, grads, 'b-', alpha=0.5, linewidth=0.5)
    if n > 50:
        s_grad = smooth(grads, min(50, n // 5))
        offset = len(grads) - len(s_grad)
        ax.plot(steps[offset:], s_grad, 'r-', linewidth=1.5)
    ax.set_yscale('log')
    ax.set_xlabel('Step')
    ax.set_ylabel('Grad Norm (log)')
    ax.set_title(f'Grad Norm (latest={grads[-1]:.1f})')
    ax.grid(True, alpha=0.3)

    # Plot 4: LR schedule + Stats
    ax = axes[1, 1]
    ax2 = ax.twinx()
    ax.plot(steps, lrs, 'g-', linewidth=1.5, label='Learning Rate')
    ax.set_xlabel('Step')
    ax.set_ylabel('Learning Rate', color='g')
    ax.tick_params(axis='y', labelcolor='g')
    if max_steps != "?":
        pct = last_step / max_steps * 100
        ax.text(0.02, 0.95, f'Progress: {pct:.1f}% ({last_step}/{max_steps})',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    ax.grid(True, alpha=0.3)

    ax2.set_ylabel('Loss Stats', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    stats_text = (
        f"  Latest loss:  {losses[-1]:.4f}\n"
        f"  Min loss:     {min(losses):.4f}\n"
        f"  Mean loss:    {np.mean(losses):.4f}\n"
        f"  Latest LR:    {lrs[-1]:.2e}\n"
        f"  Latest grad:  {grads[-1]:.1f}\n"
    )
    ax2.text(0.02, 0.02, stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='bottom', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f'loss_step{last_step}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    return True


if __name__ == '__main__':
    # Priority 1: Check for terminal log file (2.txt)
    if os.path.exists(LOG_FILE):
        print(f"Parsing log file: {LOG_FILE}")
        steps, losses, lrs, grads = parse_log_file(LOG_FILE)
        if steps:
            print(f"Found {len(steps)} log entries")
            if generate_plot_from_log(steps, losses, lrs, grads):
                plot_name = f'loss_step{steps[-1]}.png'
                print(f"Plot saved to: {os.path.join(OUTPUT_DIR, plot_name)}")
                print(f"  Total steps: {len(steps)}")
                print(f"  Loss range:  {min(losses):.4f} - {max(losses):.4f}")
                print(f"  Final loss:  {losses[-1]:.4f}")
                sys.exit(0)
            else:
                print("Not enough data points (need at least 2)")
                sys.exit(1)
        else:
            print("No valid log entries found")

    # Priority 2: Check for checkpoint trainer_state.json
    if os.path.exists(OUTPUT_DIR) and any(d.startswith("checkpoint-") for d in os.listdir(OUTPUT_DIR)):
        state = get_trainer_state()
        if state:
            print(f"Reading from checkpoint trainer_state.json")
            if generate_plot_from_state(state, 0):
                steps, losses, _, _ = extract_data(state)
                print(f"Plot saved to: {os.path.join(OUTPUT_DIR, 'loss_step' + str(steps[-1]) + '.png')}")
                sys.exit(0)

    print("Error: No training data found. Provide a log file (2.txt) or checkpoint directory.")
    sys.exit(1)
