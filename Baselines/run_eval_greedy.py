import os
import time
import torch
import numpy as np

import greedy_tsp  # compiled extension: greedy_tsp.greedy_tsp_nn


def eval_greedy_on_tsp_pt(dataset_path: str, start: int = 0, max_instances: int | None = None):
    """
    Reports:
      - cost per instance
      - runtime per instance
      - mean cost
      - std cost
      - mean runtime
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    data = torch.load(dataset_path, map_location="cpu")
    if "coordinates" not in data:
        raise KeyError("Expected a dict with key 'coordinates' in the .pt file")

    coords = data["coordinates"]  # expected shape: (B, N, 2)
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().numpy()

    if coords.ndim != 3 or coords.shape[-1] != 2:
        raise ValueError(f"Expected coordinates shape (B, N, 2), got {coords.shape}")

    B, N, _ = coords.shape
    if max_instances is not None:
        B = min(B, max_instances)
        coords = coords[:B]

    # ensure float64 for stable distance computations (you can use float32 if you want speed)
    coords = np.asarray(coords, dtype=np.float64)

    costs = np.empty(B, dtype=np.float64)
    runtimes = np.empty(B, dtype=np.float64)

    # Evaluate per instance
    for i in range(B):
        xy = coords[i]  # (N,2)
        t0 = time.perf_counter()
        tour, length = greedy_tsp.greedy_tsp_nn(xy, start=start)
        t1 = time.perf_counter()

        costs[i] = float(length)
        runtimes[i] = (t1 - t0)

        # per-instance report (keep it lightweight; comment out if you don't want printing)
        print(f"[{i+1:4d}/{B}] cost={costs[i]:.6f}  time={runtimes[i]*1000:.3f} ms")

    # Summary stats
    mean_cost = float(costs.mean())
    std_cost = float(costs.std(ddof=1)) if B > 1 else 0.0

    mean_rt = float(runtimes.mean())
    std_rt = float(runtimes.std(ddof=1)) if B > 1 else 0.0

    print("\n" + "=" * 60)
    print(f"Dataset: {dataset_path}")
    print(f"Instances evaluated: {B}   N={N}   start={start}")
    print("-" * 60)
    print(f"Mean cost     : {mean_cost:.6f}")
    print(f"Std  cost     : {std_cost:.6f}")
    print(f"Mean runtime  : {mean_rt*1000:.3f} ms / instance")
    print(f"Std  runtime  : {std_rt*1000:.3f} ms / instance")
    print("=" * 60)

    return costs, runtimes, mean_cost, std_cost, mean_rt, std_rt


if __name__ == "__main__":
    dataset_path = "../data/tsp_500_uniform_test.pt"
    # max_instances = 100  # optional quick test
    max_instances = None

    eval_greedy_on_tsp_pt(
        dataset_path=dataset_path,
        start=0,
        max_instances=max_instances,
    )

