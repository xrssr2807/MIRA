#!/usr/bin/env python3
"""
MIRA 训练 Loss 图生成器
直接从 checkpoint 的 trainer_state.json 读取数据，每 N 步生成一张 loss 图
"""
import os
import sys
import time
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = "/home/zjl/MIRA/ppg_output"
INTERVAL = 500  # Generate plot every N steps


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


def generate_plot(state, step_number):
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
    print(f"MIRA Loss Plot Monitor - checking every {INTERVAL} steps")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    last_plot_step = 0

    while True:
        state = get_trainer_state()
        if state is None:
            print(f"\r[{time.strftime('%H:%M:%S')}] 等待 checkpoint 出现...", end="", flush=True)
            time.sleep(15)
            continue

        # Clear waiting line
        print("\r" + " " * 60 + "\r", end="")

        steps, losses, lrs, grads = extract_data(state)
        if not steps:
            time.sleep(15)
            continue

        current_step = steps[-1]
        next_milestone = ((current_step // INTERVAL) + 1) * INTERVAL

        if current_step >= next_milestone and next_milestone != last_plot_step:
            if generate_plot(state, next_milestone):
                print(f"[{time.strftime('%H:%M:%S')}] ✅ Plot generated: loss_step{next_milestone}.png")
                print(f"       Step {next_milestone}/{state.get('max_steps','?')} | "
                      f"Loss={losses[-1]:.4f} | LR={lrs[-1]:.2e}")
                last_plot_step = next_milestone
        else:
            if current_step != last_plot_step:
                max_steps = state.get("max_steps", "?")
                pct = current_step / max_steps * 100 if max_steps != "?" else 0
                print(f"\r[{time.strftime('%H:%M:%S')}] Step {current_step}/{max_steps} ({pct:.1f}%) | "
                      f"Loss={losses[-1]:.4f} | LR={lrs[-1]:.2e} | "
                      f"Next plot at step {next_milestone}   ", end="", flush=True)
                last_plot_step = current_step

        time.sleep(15)
