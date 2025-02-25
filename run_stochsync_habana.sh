#!/bin/bash

prompts=(
    "Desert landscape with vast stretches of sand dunes"
    "Abandoned factory with soft rays through dusty air, eerie stillness"
    "Minimalist workspace with a soft glow, peaceful mood"
    "Graffiti-covered alleyway with street art murals"
)

steps=(10 30 50 70)

for step in "${steps[@]}"; do
    save_dir="./outputs/stochsync_final_${step}"
    for prompt in "${prompts[@]}"; do
        python run_stochsync.py \
            --model_id "stabilityai/stable-diffusion-2-1-base" \
            --prompt "$prompt" \
            --num_inner_steps "$step" \
            --save_dir "$save_dir" \
            --num_images 3 \
            --compute "hpu" \
            --seed 0
    done
done
