#!/bin/bash

model_size="small"
loss_threshold=0.1
max_tokens_per_phrase=20
top_p=0.9
max_width=128

# This if-else is to make so that max_width * (width_decay_factor ^ max_depth) is around 2.0
for max_depth in 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32; do
    # Set width_decay_factor based on max_depth
    if [ "$max_depth" -eq 4 ]; then
        width_decay_factor=0.36
    elif [ "$max_depth" -eq 5 ]; then
        width_decay_factor=0.44
    elif [ "$max_depth" -eq 6 ]; then
        width_decay_factor=0.5
    elif [ "$max_depth" -eq 7 ]; then
        width_decay_factor=0.56
    elif [ "$max_depth" -eq 8 ]; then
        width_decay_factor=0.6
    elif [ "$max_depth" -eq 9 ]; then
        width_decay_factor=0.63
    elif [ "$max_depth" -eq 10 ]; then
        width_decay_factor=0.66
    elif [ "$max_depth" -eq 11 ]; then
        width_decay_factor=0.69
    elif [ "$max_depth" -eq 12 ]; then
        width_decay_factor=0.71
    elif [ "$max_depth" -eq 13 ]; then
        width_decay_factor=0.73
    elif [ "$max_depth" -eq 14 ]; then
        width_decay_factor=0.75
    elif [ "$max_depth" -eq 15 ]; then
        width_decay_factor=0.76
    elif [ "$max_depth" -eq 16 ]; then
        width_decay_factor=0.78
    elif [ "$max_depth" -eq 17 ]; then
        width_decay_factor=0.79
    elif [ "$max_depth" -eq 18 ]; then
        width_decay_factor=0.80
    elif [ "$max_depth" -eq 19 ]; then
        width_decay_factor=0.81
    elif [ "$max_depth" -eq 20 ]; then
        width_decay_factor=0.82
    elif [ "$max_depth" -eq 21 ]; then
        width_decay_factor=0.83
    elif [ "$max_depth" -eq 22 ]; then
        width_decay_factor=0.83
    elif [ "$max_depth" -eq 23 ]; then
        width_decay_factor=0.84
    elif [ "$max_depth" -eq 24 ]; then
        width_decay_factor=0.85
    elif [ "$max_depth" -eq 25 ]; then
        width_decay_factor=0.85
    elif [ "$max_depth" -eq 26 ]; then
        width_decay_factor=0.86
    elif [ "$max_depth" -eq 27 ]; then
        width_decay_factor=0.86
    elif [ "$max_depth" -eq 28 ]; then
        width_decay_factor=0.87
    elif [ "$max_depth" -eq 29 ]; then
        width_decay_factor=0.87
    elif [ "$max_depth" -eq 30 ]; then
        width_decay_factor=0.88
    elif [ "$max_depth" -eq 31 ]; then
        width_decay_factor=0.88
    elif [ "$max_depth" -eq 32 ]; then
        width_decay_factor=0.88
    else
        echo "No mapping defined for max_depth=$max_depth!"
        exit 1
    fi

    # You can adjust or add more values if needed:
    for num_classes in 10000 1000 100; do
        for allowed_threshold in 0.5; do
            for temperature in 0.75; do
                # Construct the expected JSON filename
                json_file="trees/model_size_${model_size}_num_classes_${num_classes}\
_allowed_threshold_${allowed_threshold}_loss_threshold_${loss_threshold}_top_p_${top_p}\
_max_depth_${max_depth}_max_width_${max_width}\
_temperature_${temperature}\
_width_decay_factor_${width_decay_factor}.json"

                # Check if JSON already exists
                if [ -f "$json_file" ]; then
                    echo "JSON already exists: $json_file. Skipping generation."
                    continue
                fi

                # Run the Python script with the chosen parameters
                python build_taxonomy.py \
                    --tokenizer_path "./custom_tokenizer" \
                    --num_classes "$num_classes" \
                    --force_device cuda \
                    --top_p "$top_p" \
                    --max_depth "$max_depth" \
                    --max_width "$max_width" \
                    --max_tokens_per_phrase "$max_tokens_per_phrase" \
                    --temperature "$temperature" \
                    --allowed_threshold "$allowed_threshold" \
                    --loss_threshold "$loss_threshold" \
                    --model_size "$model_size" \
                    --width_decay_factor "$width_decay_factor"

            done
        done
    done
done
