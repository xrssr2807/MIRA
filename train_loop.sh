#!/bin/bash
# Auto-restart training loop - resumes from latest checkpoint on crash

OUTPUT_DIR="ppg_output"
MAX_RESTARTS=10
RESTART_COUNT=0

# Improved hyperparameters for stable convergence
LEARNING_RATE="3e-5"
WARMUP_RATIO="0.1"
ADAM_BETA2="0.999"
WEIGHT_DECAY="0.01"
SAVE_STEPS=100

while [ $RESTART_COUNT -lt $MAX_RESTARTS ]; do
    # Find latest checkpoint
    CHECKPOINT=$(ls -dt ${OUTPUT_DIR}/checkpoint-* 2>/dev/null | head -1)

    if [ -n "$CHECKPOINT" ]; then
        MODEL_PATH="$CHECKPOINT"
        echo "[$(date)] Resuming from checkpoint: $MODEL_PATH"
    else
        MODEL_PATH="Maple728/TimeMoE-50M"
        echo "[$(date)] No checkpoint found. Starting from scratch."
    fi

    FROM_SCRATCH_FLAG=""
    if [ -z "$CHECKPOINT" ]; then
        FROM_SCRATCH_FLAG="--from_scratch"
    fi

    echo "[$(date)] Starting training (restart #$RESTART_COUNT)..."

    torchrun --master_addr=localhost --master_port=9899 \
        --node_rank=0 --nproc_per_node=1 --nnodes=1 \
        main.py \
        --model_path "$MODEL_PATH" \
        $FROM_SCRATCH_FLAG \
        --data_path ppg_full \
        --output_path "$OUTPUT_DIR" \
        --max_length 1024 \
        --normalization_method zero \
        --attn_implementation eager \
        --micro_batch_size 52 \
        --global_batch_size 52 \
        --precision bf16 \
        --save_strategy steps \
        --save_steps $SAVE_STEPS \
        --gradient_checkpointing \
        --save_only_model \
        --save_total_limit 5 \
        --logging_steps 10 \
        --lr_scheduler_type cosine \
        --warmup_ratio $WARMUP_RATIO \
        --warmup_start_lr 1e-4 \
        --weight_decay $WEIGHT_DECAY \
        --learning_rate $LEARNING_RATE \
        --num_train_epochs 1.0 \
        --dataloader_num_workers 8 \
        --adam_beta2 $ADAM_BETA2

    EXIT_CODE=$?
    echo "[$(date)] Training exited with code: $EXIT_CODE"

    # Check if training completed normally (not crash)
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date)] Training completed successfully!"
        break
    fi

    RESTART_COUNT=$((RESTART_COUNT + 1))
    echo "[$(date)] Will restart in 5 seconds... (attempt $RESTART_COUNT/$MAX_RESTARTS)"
    sleep 5
done

if [ $RESTART_COUNT -ge $MAX_RESTARTS ]; then
    echo "[$(date)] ERROR: Max restart attempts ($MAX_RESTARTS) reached. Giving up."
fi
