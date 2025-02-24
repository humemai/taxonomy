#!/bin/bash

# Loop over num_classes values
for classes in 10 100 1000 10000; do
    # Loop over allowed_threshold values
    for threshold in 0.7 0.6 0.5 0.4 0.3; do
        echo "Running with allowed_threshold=${threshold} and num_classes=${classes}"
        python train.py \
            --num_workers 8 \
            --max_length 256 \
            --allowed_threshold ${threshold} \
            --per_device_train_batch_size 64 \
            --logging_steps 50 \
            --save_steps 250 \
            --max_steps 50000 \
            --loss_threshold 0.1 \
            --save_total_limit 2 \
            --num_classes ${classes} \
            --sampling_mode class_aware \
            --model_size small
    done
done
