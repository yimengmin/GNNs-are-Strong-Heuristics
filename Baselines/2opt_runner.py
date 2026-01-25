import os
import math
import time
from typing import List, Tuple

import torch
import random
from statistics import mean, stdev


"""
Reports:
- cost per instance
- runtime per instance
- mean cost
- std cost
- mean runtime
"""


############################################################
# Basic distance utilities
############################################################

def euclidean_distance(a, b) -> float:
    dx = float(a[0] - b[0])
    dy = float(a[1] - b[1])
    return math.sqrt(dx * dx + dy * dy)


def tour_length(coords, tour: List[int]) -> float:
    n = len(tour)
    total = 0.0
    for i in range(n):
        a = coords[tour[i]]
        b = coords[tour[(i + 1) % n]]
        total += euclidean_distance(a, b)
    return total


############################################################
# Nearest Neighbor Initial Tour
############################################################

def nearest_neighbor_tour(coords) -> List[int]:
    n = coords.shape[0]
    unvisited = set(range(n))
    start = 0
    tour = [start]
    unvisited.remove(start)
    current = start

    while unvisited:
        best_next = None
        best_dist = float("inf")
        for j in unvisited:
            d = euclidean_distance(coords[current], coords[j])
            if d < best_dist:
                best_dist = d
                best_next = j
        tour.append(best_next)
        unvisited.remove(best_next)
        current = best_next

    return tour


############################################################
# 2-OPT (no max passes, loop until no improvement)
############################################################

def two_opt(coords, initial_tour=None, epsilon=1e-12) -> Tuple[List[int], float]:
    n = coords.shape[0]

    if initial_tour is None:
        tour = list(range(n))
        random.shuffle(tour)
    elif initial_tour == "nn":
        tour = nearest_neighbor_tour(coords)
    else:
        tour = list(initial_tour)

    best_len = tour_length(coords, tour)

    while True:
        improved = False

        for i in range(1, n - 2):
            for k in range(i + 1, n - 1):
                a, b = tour[i - 1], tour[i]
                c, d = tour[k], tour[(k + 1) % n]

                delta = (
                    euclidean_distance(coords[a], coords[c]) +
                    euclidean_distance(coords[b], coords[d]) -
                    euclidean_distance(coords[a], coords[b]) -
                    euclidean_distance(coords[c], coords[d])
                )

                if delta < -epsilon:
                    tour[i:k+1] = reversed(tour[i:k+1])
                    best_len += delta
                    improved = True

        if not improved:
            break

    return tour, best_len


############################################################
# Dataset processing
############################################################

def coords_to_complete_graph_data(coords):
    # for 2-opt we only need coords
    return {"coords": coords}


def load_and_process_tsp_dataset(filepath):
    print(f"Loading dataset from: {filepath}")

    data = torch.load(filepath, map_location="cpu")
    coordinates = data["coordinates"]
    num_instances, num_cities = coordinates.shape[:2]

    print(f"Loaded {num_instances} instances with {num_cities} cities each")

    data_list = []
    for i in range(num_instances):
        item = coords_to_complete_graph_data(coordinates[i])
        data_list.append(item)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{num_instances} instances")

    print("Dataset processing complete!")
    return data_list


############################################################
# 2-OPT baseline + timing
############################################################

def two_opt_baseline(data_list, init="nn", verbose=True):
    """
    Reports:
    - cost per instance
    - runtime per instance
    - mean cost
    - std cost
    - mean runtime

    Args:
        data_list: list of {"coords": tensor (N,2)}
        init: None | "nn" | permutation list
        verbose: print per-instance lines
    Returns:
        mean_cost, std_cost, mean_runtime, std_runtime, costs, runtimes
    """
    costs = []
    runtimes = []

    if verbose:
        print("\nProcessing each instance with 2-opt:")
        print("-" * 70)

    for idx, item in enumerate(data_list):
        coords = item["coords"]

        t0 = time.perf_counter()
        _, cost = two_opt(coords, initial_tour=init)
        t1 = time.perf_counter()

        rt = t1 - t0
        costs.append(float(cost))
        runtimes.append(float(rt))

        if verbose:
            print(
                f"Instance {idx + 1:4d}/{len(data_list)}: "
                f"cost = {cost:.4f} | time = {rt*1000:.3f} ms"
            )

    mean_cost = mean(costs)
    std_cost = stdev(costs) if len(costs) > 1 else 0.0

    mean_rt = mean(runtimes)
    std_rt = stdev(runtimes) if len(runtimes) > 1 else 0.0

    return mean_cost, std_cost, mean_rt, std_rt, costs, runtimes


############################################################
# Main
############################################################

def main():
    dataset_path = "../data/tsp_500_uniform_test.pt"

    if not os.path.exists(dataset_path):
        print(f"Dataset file not found: {dataset_path}")
        return

    data_list = load_and_process_tsp_dataset(dataset_path)

    print("\n" + "=" * 60)
    print("Running 2-OPT Baseline")
    print("=" * 60)

    # Test subset
    test_subset = data_list[:10]
    print(f"\nTesting on first {len(test_subset)} instances...")

    mean_cost, std_cost, mean_rt, std_rt, _, _ = two_opt_baseline(
        test_subset, init="nn", verbose=True
    )

    print("\n[Subset Results]")
    print(f"Mean cost    : {mean_cost:.4f}")
    print(f"Std  cost    : {std_cost:.4f}")
    print(f"Mean runtime : {mean_rt*1000:.3f} ms / instance")
    print(f"Std  runtime : {std_rt*1000:.3f} ms / instance")

    # Full dataset
    print(f"\nRunning on full dataset ({len(data_list)} instances)...")
    mean_cost, std_cost, mean_rt, std_rt, _, _ = two_opt_baseline(
        data_list, init="nn", verbose=False
    )

    print("\n[Full Dataset Results]")
    print(f"Mean cost    : {mean_cost:.4f}")
    print(f"Std  cost    : {std_cost:.4f}")
    print(f"Mean runtime : {mean_rt*1000:.3f} ms / instance")
    print(f"Std  runtime : {std_rt*1000:.3f} ms / instance")


if __name__ == "__main__":
    main()


