#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MIRA Model Fine-tuning Script — RTX 4090 24GB Optimized.

Goal: High GPU compute & memory utilization on consumer GPU.

Key optimizations:
1. micro_batch_size tuned for 24GB VRAM
2. BF16 for Tensor Core acceleration
3. Gradient checkpointing to trade compute for memory
4. Adequate dataloader workers to keep GPU fed
5. cudnn benchmark enabled

Usage:
    python finetune.py
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mira.runner import MIRARunner

# ============================================================
#  Configuration — 修改这里的值适配你的下游任务
# ============================================================

# 预训练模型路径（包含 model.safetensors + config.json）
MODEL_PATH = "/root/model"

# 微调输出路径
OUTPUT_PATH = "ppg_output/finetune_v1"

# 下游微调数据路径（PKL 文件目录 / JSONL 文件 / memmap 目录）
DATA_PATH = "/root/processed_dataset"

# max_length 越小，batch size 可以越大。
# 256 是默认值，如果你的数据允许，可以降到 128 来进一步提高 batch size。
MAX_LENGTH = 256

# 微调建议比预训练小的学习率，避免破坏已有知识。
LEARNING_RATE = 1e-5
NUM_EPOCHS = 3

# ============================================================
#  RTX 4090 24GB 优化参数
# ============================================================

# --- Batch Size 计算 ---
# RTX 4090 24GB + BF16 + max_length=256 + gradient_checkpointing:
#   模型权重 ~1GB (bf16)
#   Adam 优化器状态 ~2GB (fp32 master weights + 动量 + 方差)
#   剩余 ~21GB 给 activations (gradient checkpointing 大幅降低)
#   对于 32-layer, hidden=4096 的模型:
#     max_length=256 + checkpointing → micro_batch_size=32 大约占 12-16GB 显存
#     gradient_accumulation=16 → global_batch_size=512
# 如果 max_length=128，micro_batch_size 可以推到 64

MICRO_BATCH_SIZE = 32           # 单卡 micro batch，占满显存的关键参数
GRADIENT_ACCUMULATION = 16      # 梯度累积，global = micro * accum * num_devices
GLOBAL_BATCH_SIZE = MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION  # 512 (单卡)

# 如果你想测试最大 batch size，取消注释下面这行来自动调优:
# MICRO_BATCH_SIZE = auto_tune_batch_size(MODEL_PATH, MAX_LENGTH)

TRAIN_CONFIG = {
    # --- 数据 ---
    "data_path": DATA_PATH,
    "max_length": MAX_LENGTH,
    "normalization_method": "zero",
    "time_aware_dataset": True,
    "time_normalization": "none",
    "data_sample_size": 1000,

    # --- 训练步数 / 轮数（二选一）---
    "num_train_epochs": NUM_EPOCHS,
    "train_steps": None,

    # --- 学习率 ---
    "learning_rate": LEARNING_RATE,
    "min_learning_rate": 1e-7,
    "warmup_ratio": 0.05,
    "warmup_start_lr": 1e-6,

    # --- Batch Size ---
    "global_batch_size": GLOBAL_BATCH_SIZE,
    "micro_batch_size": MICRO_BATCH_SIZE,

    # --- 精度 ---
    "precision": "bf16",

    # --- 优化器 ---
    "optim": "adamw_torch",
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
    "adam_epsilon": 1e-8,
    "max_grad_norm": 1.0,
    "weight_decay": 0.1,

    # --- 调度器 ---
    "lr_scheduler_type": "cosine",

    # --- 保存策略 ---
    "save_strategy": "steps",
    "save_steps": 200,
    "save_total_limit": 3,
    "save_only_model": True,

    # --- 日志 ---
    "logging_steps": 5,

    # --- 性能优化 (关键!) ---
    "gradient_checkpointing": True,   # 用计算换显存 → 允许更大的 batch size
    "torch_compile": False,           # RTX 4090 上 torch.compile 可能不稳定
    "dataloader_num_workers": 4,      # 多进程预加载，保持 GPU 不空闲
    "ddp_find_unused_parameters": True,

    # --- 注意力 ---
    "attn_implementation": "eager",   # CT-RoPE 必须用 eager
}


def print_gpu_info():
    """Print RTX 4090 GPU info before training."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_mem / 1e9
        print("=" * 60)
        print(f"  GPU: {props.name}")
        print(f"  VRAM: {vram_gb:.1f} GB")
        print(f"  Compute: {props.major}.{props.minor}")
        print(f"  SMs: {props.multi_processor_count}")
        print(f"  Micro batch: {MICRO_BATCH_SIZE}")
        print(f"  Global batch: {GLOBAL_BATCH_SIZE}")
        print(f"  Max length: {MAX_LENGTH}")
        print(f"  Precision: BF16")
        print(f"  Gradient checkpointing: {TRAIN_CONFIG['gradient_checkpointing']}")
        print(f"  torch.compile: {TRAIN_CONFIG['torch_compile']}")
        print("=" * 60)


def main():
    print_gpu_info()

    print(f"\n[INFO] Fine-tuning model: {MODEL_PATH}")
    print(f"[INFO] Data path: {DATA_PATH}")
    print(f"[INFO] Output path: {OUTPUT_PATH}")
    print(f"[INFO] Epochs: {NUM_EPOCHS}, LR: {LEARNING_RATE}")

    runner = MIRARunner(
        model_path=MODEL_PATH,
        output_path=OUTPUT_PATH,
        seed=9899,
    )

    # from_scratch=False 加载已有 safetensors 权重继续训练
    runner.train_model(
        from_scratch=False,
        resume_from_checkpoint=None,
        **TRAIN_CONFIG,
    )

    print(f"\n[INFO] Fine-tuning complete. Model saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
