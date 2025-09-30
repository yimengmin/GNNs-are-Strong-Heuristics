python test_ddp.py   --test_data ./data/tsp_100_uniform_test.pt --save_dir test_results_shift  --num_nodes 100  --model_path SaveModels/S100NIter100NL16EPS1000Tau3.0HID256OPTadamDS5.0DP0.10/ddpbest_size_100_hidden_256_adam_tau_3.0_n_iter_100_noise_0.1_shift_-1_dist_scale_5.0_n_layers16_seed_42_dropout0.1.ckpt


python test_ddp.py   --test_data ./data/tsp_200_uniform_test.pt --save_dir test_results_shift  --num_nodes 200  --model_path  SaveModels/S200NIter100NL20EPS1000Tau3.0HID256OPTadamDS5.0DP0.10/ddpbest_size_200_hidden_256_adam_tau_3.0_n_iter_100_noise_0.1_shift_-1_dist_scale_5.0_n_layers20_seed_42_dropout0.1.ckpt


python test_ddp.py   --test_data ./data/tsp_500_uniform_test.pt --save_dir test_results_shift  --num_nodes 500  --model_path SaveModels/S500NIter120NL24EPS1000Tau3.0HID384OPTadamDS5.0DP0.10/ddpbest_size_500_hidden_384_adam_tau_3.0_n_iter_120_noise_0.1_shift_-1_dist_scale_5.0_n_layers24_seed_42_dropout0.1.ckpt  --compute_greedy

