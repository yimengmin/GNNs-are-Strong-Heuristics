# Graph Neural Networks are Heuristics

## Installation & Dependencies

- ### Required Dependencies
Before running the code, ensure you have installed the torch-linear-assignment: https://www.piwheels.org/project/torch-linear-assignment. I am using Version 0.0.3.

- ### Required Dependencies

For TSP-500 experiments, I use float16 (bf16) precision. All performance results are evaluated on NVIDIA A100 GPUs, as using other GPU architectures may lead to numerical or performance inconsistencies.


##  Run Inference:

- TSP 100
```
python test_ddp.py   --test_data ./data/tsp_100_uniform_test.pt --save_dir test_results_shift  --num_nodes 100  --model_path SaveModels/S100NIter100NL16EPS1000Tau3.0HID256OPTadamDS5.0DP0.10/ddpbest_size_100_hidden_256_adam_tau_3.0_n_iter_100_noise_0.1_shift_-1_dist_scale_5.0_n_layers16_seed_42_dropout0.1.ckpt --batch_size 512

```
- TSP 200
```
python test_ddp.py   --test_data ./data/tsp_200_uniform_test.pt --save_dir test_results_shift  --num_nodes 200  --model_path  SaveModels/S200NIter100NL24EPS1000Tau3.0HID256OPTadamDS5.0DP0.10/ddpbest_size_200_hidden_256_adam_tau_3.0_n_iter_100_noise_0.1_shift_-1_dist_scale_5.0_n_layers24_seed_42_dropout0.1.ckpt  --batch_size 512
```

To enable MC dropout during evaluation, edit `test_ddp.py` and set the following flag at line 312:
```python
enable_mc_dropout = True
```



- TSP 500
```
python test_ddp_bf16.py   --test_data ./data/tsp_500_uniform_test.pt --save_dir test_results_shift  --num_nodes 500  --model_path SaveModels/BF16S500NIter100NL48EPS1000Tau3.5HID384OPTadamDS5.0DP0.5/ddpbest_size_500_hidden_384_adam_tau_3.5_n_iter_100_noise_0.1_shift_-1_dist_scale_5.0_n_layers48_seed_42_dropout0.5.ckpt --batch_size 512
```

To enable MC dropout during evaluation, edit `test_ddp_bf16.py` and set the following flag at line 340:
```python
enable_mc_dropout = True
```


## Ensembling

go to test_results_shift/

and run 
```
compare_results.py file1 file2 ...
```
