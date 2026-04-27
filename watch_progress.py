#!/usr/bin/env python3
"""
MIRA 训练进度监控器
实时显示训练进度和预计剩余时间
通过追踪 trainer_state.json 的修改时间来计算速度
"""
import os
import sys
import time
import json
from datetime import datetime

OUTPUT_DIR = "/home/zjl/MIRA/ppg_output"
REFRESH_INTERVAL = 5  # seconds


def get_progress():
    """Read training progress from trainer_state.json"""
    checkpoints = sorted([
        d for d in os.listdir(OUTPUT_DIR)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(OUTPUT_DIR, d))
    ], key=lambda x: int(x.split("-")[-1]))

    if not checkpoints:
        return None

    latest = checkpoints[-1]
    state_path = os.path.join(OUTPUT_DIR, latest, "trainer_state.json")

    if not os.path.exists(state_path):
        return None

    with open(state_path) as f:
        state = json.load(f)

    return {
        "global_step": state.get("global_step", 0),
        "max_steps": state.get("max_steps", 67926),
        "epoch": state.get("epoch", 0),
        "checkpoint": latest,
        "log_history": state.get("log_history", []),
        "state_file_time": os.path.getmtime(state_path),
    }


def format_time(seconds):
    """Format seconds to human readable"""
    if seconds < 60:
        return f"{seconds:.0f}秒"
    elif seconds < 3600:
        return f"{seconds/60:.1f}分钟"
    elif seconds < 86400:
        h = seconds / 3600
        return f"{h:.1f}小时"
    else:
        d = seconds / 86400
        h = (seconds % 86400) / 3600
        return f"{d:.1f}天{h:.0f}小时"


def watch():
    """Watch training progress in real-time"""
    print("等待训练启动...")
    print()

    # Track file modification time to calculate speed across checkpoints
    prev_file_mtime = None
    prev_step = None
    speed = None  # seconds per step
    monitor_start = time.time()

    while True:
        progress = get_progress()

        if progress is None:
            elapsed = time.time() - monitor_start
            print(f"\r[{datetime.now().strftime('%H:%M:%S')}] 等待训练启动... (已等待 {int(elapsed)}秒)", end="", flush=True)
            time.sleep(REFRESH_INTERVAL)
            continue

        current_step = progress["global_step"]
        max_steps = progress["max_steps"]
        file_mtime = progress["state_file_time"]

        # Calculate speed when checkpoint file is updated
        if prev_file_mtime is not None and file_mtime != prev_file_mtime and current_step > prev_step:
            step_delta = current_step - prev_step
            time_delta = file_mtime - prev_file_mtime
            if time_delta > 0:
                speed = time_delta / step_delta  # sec/step

        prev_file_mtime = file_mtime
        prev_step = current_step

        # ETA
        if speed and current_step > 0:
            remaining_steps = max_steps - current_step
            eta_seconds = remaining_steps * speed
            eta_str = format_time(eta_seconds)
            speed_str = f"{speed:.1f}秒/步"
        else:
            # Fallback: estimate from wall clock since monitor start
            elapsed = time.time() - monitor_start
            if current_step > 0:
                rough_speed = elapsed / current_step
                eta_seconds = (max_steps - current_step) * rough_speed
                eta_str = f"~{format_time(eta_seconds)}"
                speed_str = f"~{rough_speed:.1f}秒/步"
            else:
                eta_str = "计算中..."
                speed_str = "计算中..."

        # Progress bar
        pct = current_step / max_steps * 100 if max_steps > 0 else 0
        bar_width = 50
        filled = int(bar_width * current_step / max_steps) if max_steps > 0 else 0
        bar = "█" * filled + "░" * (bar_width - filled)

        # Latest metrics
        loss = "?"
        lr = "?"
        grad_norm = "?"
        if progress["log_history"]:
            latest = progress["log_history"][-1]
            if "loss" in latest:
                loss = f"{latest['loss']:.4f}"
            if "learning_rate" in latest:
                lr = f"{latest['learning_rate']:.2e}"
            if "grad_norm" in latest:
                grad_norm = f"{latest['grad_norm']:.1f}"

        epoch = progress["epoch"]
        checkpoint = progress["checkpoint"]
        total_elapsed = time.time() - monitor_start
        elapsed_str = format_time(total_elapsed)

        # Display
        os.system('clear')
        print(f"╔{'═'*58}╗")
        print(f"║  MIRA 训练进度监控 {' '*36}║")
        print(f"╠{'═'*58}╣")
        print(f"║  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{' '*29}║")
        print(f"║  已运行: {elapsed_str}{' '*43}║")
        print(f"╠{'═'*58}╣")
        print(f"║  进度: [{bar}] {pct:5.1f}%       ║")
        print(f"║  步数: {current_step:6,} / {max_steps:6,}                          ║")
        print(f"║  Epoch: {epoch:.4f}{' '*43}║")
        print(f"║  Checkpoint: {checkpoint}{' '*(47-len(checkpoint))}║")
        print(f"╠{'═'*58}╣")
        print(f"║  最新 Loss: {loss}          LR: {lr}      ║")
        print(f"║  梯度范数: {grad_norm}            速度: {speed_str:12}║")
        print(f"║  预计剩余: {eta_str}{' '*37}║")
        print(f"╠{'═'*58}╣")
        print(f"║  超参数: LR=3e-5 | warmup=10% | β2=0.999 | wd=0.01  ║")
        print(f"║  Batch: 52 | workers=2 | save_steps=500              ║")
        print(f"╚{'═'*58}╝")
        print()
        print("  按 Ctrl+C 退出监控（训练在后台继续运行）")
        if not speed:
            print("  (ETA 将在下一个 checkpoint 保存后计算)")

        time.sleep(REFRESH_INTERVAL)


if __name__ == "__main__":
    try:
        watch()
    except KeyboardInterrupt:
        print("\n\n监控已停止，训练仍在后台运行。")
        sys.exit(0)
