#!/usr/bin/env bash
set -Eeuo pipefail

#savedir="SaveModels/best_stable_sct_model_size_100_hidden_256_adam_tau_5.0_n_iter_100_noise_0.1_shift_-1_dist_scale_5.0_n_layers16_seed_42.pt"
savedir="SaveModels/best_stable_sct_model_size_100_hidden_256_adam_tau_3.0_n_iter_100_noise_0.1_shift_-1_dist_scale_5.0_n_layers16_seed_42.pt"

out_dir="test_results_shift"
mkdir -p "$out_dir"

for i in $(seq 1 100); do
  echo "[$(date '+%F %T')] Run $i/100 …"

  python test.py \
    --test_data "./data/tsp_100_uniform_test.pt" \
    --save_dir "$out_dir" \
    --num_nodes 100 \
    --model_path "$savedir" \

  cp "$out_dir/all_tour_lengths_shift_-1_size_100.txt" \
     "$out_dir/S100SFT-1v${i}.txt"
done

