#!/usr/bin/env python3
"""
MIRA 训练 Loss 图生成器
监控训练输出，每 1000 步生成一张 loss 图
"""
import re
import sys
import time
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = "/home/zjl/MIRA/ppg_output"
LOG_FILE = os.path.join(OUTPUT_DIR, "training_output.log")
PLOT_DIR = OUTPUT_DIR
INTERVAL = 100  # Generate plot every 100 steps

def parse_loss_from_file(filepath):
    """Parse all loss entries from training log file"""
    if not os.path.exists(filepath):
        return [], [], [], []
    with open(filepath, 'r', errors='ignore') as f:
        content = f.read()
    pattern = r"\{'loss': ([\d.]+), 'grad_norm': ([\d.e+]+), 'learning_rate': ([\d.e+-]+), 'epoch': ([\d.]+)\}"
    matches = re.findall(pattern, content)
    steps, losses, lrs, grads = [], [], [], []
    for i, (loss, grad, lr, epoch) in enumerate(matches):
        step = (i + 1) * 10  # logging_steps=10
        steps.append(step)
        losses.append(float(loss))
        lrs.append(float(lr))
        grads.append(float(grad))
    return steps, losses, lrs, grads

def smooth(y, window=20):
    """Moving average smoothing"""
    if len(y) <= window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode='valid')

def generate_plot(output_path, title_suffix=""):
    """Generate loss plot from training logs"""
    steps, losses, lrs, grads = parse_loss_from_file(LOG_FILE)
    if len(steps) < 2:
        return False
    # Also append saved plots from previous runs
    prev_steps, prev_losses, prev_lrs, prev_grads = [], [], [], []
    for f in sorted(os.listdir(PLOT_DIR)):
        if f.startswith('loss_step') and f.endswith('.npz'):
            data = np.load(os.path.join(PLOT_DIR, f))
            prev_steps.extend(data['steps'].tolist())
            prev_losses.extend(data['losses'].tolist())
            prev_lrs.extend(data['lrs'].tolist())
            prev_grads.extend(data['grads'].tolist())
    # Combine prev + current
    all_steps = prev_steps + steps
    all_losses = prev_losses + losses
    all_lrs = prev_lrs + lrs
    all_grads = prev_grads + grads
    if not all_steps:
        return False
    # Remove duplicate steps at boundary
    if prev_steps and steps:
        boundary = prev_steps[-1]
        dedup_steps = []
        dedup_losses = []
        dedup_lrs = []
        dedup_grads = []
        for i, (s, l, lr, g) in enumerate(zip(all_steps, all_losses, all_lrs, all_grads)):
            if i < len(prev_steps) or s > boundary:
                dedup_steps.append(s)
                dedup_losses.append(l)
                dedup_lrs.append(lr)
                dedup_grads.append(g)
        all_steps, all_losses, all_lrs, all_grads = dedup_steps, dedup_losses, dedup_lrs, dedup_grads
    n = len(all_steps)
    last_step = all_steps[-1]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'MIRA Training Loss - Step {last_step} {title_suffix}', fontsize=14, fontweight='bold')
    # Plot 1: Full loss with smoothing
    ax = axes[0, 0]
    ax.plot(all_steps, all_losses, 'b-', alpha=0.5, linewidth=0.5, label='Raw loss')
    if n > 50:
        s_loss = smooth(all_losses, min(50, n//5))
        offset = len(all_losses) - len(s_loss)
        ax.plot(all_steps[offset:], s_loss, 'r-', linewidth=2, label=f'Smooth (w={min(50,n//5)})')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title(f'Loss (n={n} pts)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    # Plot 2: Zoom on last 1000 steps
    ax = axes[0, 1]
    recent_mask = np.array(all_steps) > (last_step - 1000)
    if recent_mask.sum() > 1:
        r_steps = np.array(all_steps)[recent_mask]
        r_losses = np.array(all_losses)[recent_mask]
        ax.plot(r_steps, r_losses, 'b-', alpha=0.6, linewidth=0.8)
        if len(r_losses) > 20:
            s_r = smooth(r_losses, min(20, len(r_losses)//3))
            ax.plot(r_steps[-len(s_r):], s_r, 'r-', linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title(f'Last 1000 steps (mean={r_losses.mean():.4f}, std={r_losses.std():.4f})')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Not enough data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Last 1000 steps')
    # Plot 3: Gradient norm (log scale)
    ax = axes[1, 0]
    ax.plot(all_steps, all_grads, 'b-', alpha=0.5, linewidth=0.5)
    if n > 50:
        s_grad = smooth(all_grads, min(50, n//5))
        offset = len(all_grads) - len(s_grad)
        ax.plot(all_steps[offset:], s_grad, 'r-', linewidth=1.5)
    ax.set_yscale('log')
    ax.set_xlabel('Step')
    ax.set_ylabel('Grad Norm (log)')
    ax.set_title(f'Grad Norm (latest={all_grads[-1]:.4f})')
    ax.grid(True, alpha=0.3)
    # Plot 4: Stats
    ax = axes[1, 1]
    ax.axis('off')
    stats = (
        f"Training Statistics (up to step {last_step})\n"
        f"{'='*40}\n"
        f"Total logged steps:  {n}\n"
        f"Initial loss:        {all_losses[0]:.4f}\n"
        f"Latest loss:         {all_losses[-1]:.4f}\n"
        f"Min loss:            {min(all_losses):.4f}\n"
        f"Max loss:            {max(all_losses):.4f}\n"
        f"Mean loss (all):     {np.mean(all_losses):.4f}\n"
        f"Std loss (all):      {np.std(all_losses):.4f}\n"
        f"Latest LR:           {all_lrs[-1]:.2e}\n"
        f"Latest grad_norm:    {all_grads[-1]:.4f}\n"
        f"{'='*40}\n"
        f"Hyperparameters:\n"
        f"  LR: 3e-5 (cosine)\n"
        f"  Warmup: 10%\n"
        f"  Adam beta2: 0.999\n"
        f"  Weight decay: 0.01\n"
    )
    ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    plt.tight_layout()
    plot_path = os.path.join(PLOT_DIR, f'loss_step{last_step}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    # Save data for next plot
    np.savez(os.path.join(PLOT_DIR, f'loss_step{last_step}.npz'),
             steps=np.array(all_steps), losses=np.array(all_losses),
             lrs=np.array(all_lrs), grads=np.array(all_grads))
    return True

if __name__ == '__main__':
    last_plot_step = 0
    while True:
        steps, losses, _, _ = parse_loss_from_file(LOG_FILE)
        if steps:
            current_step = steps[-1]
            # Find the next milestone
            next_milestone = ((current_step // INTERVAL) + 1) * INTERVAL
            if current_step >= next_milestone and next_milestone != last_plot_step:
                if generate_plot(f"({next_milestone} steps)"):
                    print(f"[{time.strftime('%H:%M:%S')}] Plot generated: loss_step{next_milestone}.png")
                    last_plot_step = next_milestone
        time.sleep(15)
