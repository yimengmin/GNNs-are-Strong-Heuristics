#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for trained SCT TSP model (DDP-friendly)
- Loads best/last checkpoints produced by the new trainer
- Uses data_generator_ddp (build_dataloader / SimpleTSPDataset)
- Infers output_dim from TEST DATA (robust), or --num_nodes override
- Accepts overrides for shift / distance_scale / noise_scale
- Computes greedy baseline (optional) and saves full tour-length list
"""

import argparse
import os
import time
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

from cyclefastsags_stable import EnhancedUltraFastSCTGNN as GNN
from data_generator_ddp import (
    SimpleTSPDataset,
    build_dataloader,
    load_tsp_dataset,
)
from hardpermutation import to_exact_permutation_batched
from utsploss import tsp_permutation_loss

# Optional viz helpers (guarded import)
try:
    from tsp_visualization import visualize_batch_tours  # noqa: F401
    VIZ_AVAILABLE = True
except Exception:
    VIZ_AVAILABLE = False


# ----------------------------- helpers -----------------------------

def _to_namespace(args_like):
    """Convert dict or argparse.Namespace to SimpleNamespace for getattr()."""
    if isinstance(args_like, dict):
        return SimpleNamespace(**args_like)
    return args_like  # assume has attributes already


def enable_dropout_only(model: torch.nn.Module):
    """Enable Dropout layers during eval (MC-dropout style)."""
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


def load_model(model_path: str, device: torch.device, output_dim: int) -> tuple[torch.nn.Module, SimpleNamespace]:
    """Load trained model (weights + saved args) with a specified output_dim."""
    print(f"Loading model from: {model_path}")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    # Trainer saved 'args' as a dict (vars(args)); keep backward compatibility.
    saved_args = _to_namespace(ckpt.get("args", {}))

    # Epoch / best val length (key name differs across versions)
    epoch_trained = int(ckpt.get("epoch", -1)) + 1
    best_val = ckpt.get("best_val_length", ckpt.get("val_length", float("nan")))
    print(f"Model trained for {epoch_trained} epochs")
    if best_val == best_val:  # not NaN
        print(f"Best validation length: {best_val:.2f}")

    # Create model with same architecture (but output_dim from test data or CLI)
    model = GNN(
        input_dim=2,
        hidden_dim=getattr(saved_args, "hidden_dim", 512),
        output_dim=output_dim,
        n_layers=getattr(saved_args, "n_layers", 8),
        sct_order=getattr(saved_args, "sct_order", 4),
        gcn_order=getattr(saved_args, "gcn_order", 2),
        tanh_scale=getattr(saved_args, "tanh_scale", 40.0),
        tau=getattr(saved_args, "tau", 5.0),
        n_iter=getattr(saved_args, "n_iter", 60),
        noise_scale=getattr(saved_args, "noise_scale", 0.1),
        inference_mode=True,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print(f"Model output dimension (cities): {output_dim}")

    return model, saved_args


@torch.no_grad()
def evaluate_model(model, dataloader, device, shift=-1, distance_scale=5.0, enable_mc_dropout=True, verbose=True):
    """
    Evaluate model on test data (returns dict of metrics and a first-batch sample for viz).
    """
    model.eval()
    if enable_mc_dropout:
        enable_dropout_only(model)

    all_tour_lengths = []
    sample_coordinates = None
    sample_heatmaps = None
    inference_times = []
    total_instances = 0

    for bidx, coordinates_batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        coordinates_batch = coordinates_batch.to(device, non_blocking=True)
        B, N, _ = coordinates_batch.shape

        # Pairwise distances & adjacency
        dist_mat = torch.cdist(coordinates_batch, coordinates_batch)
        adj = torch.exp(-dist_mat / float(distance_scale))

        # Forward (inference returns logits)
        t0 = time.time()
        logits = model(coordinates_batch, adj)
        P_hard = to_exact_permutation_batched(logits)
        inference_times.append(time.time() - t0)

        # Tour lengths via the same loss utility
        tour_lengths, heatmaps = tsp_permutation_loss(P_hard, dist_mat, shift)

        # Accumulate
        all_tour_lengths.extend(tour_lengths.detach().cpu().numpy().tolist())
        total_instances += B

        # Keep first batch for visualization later
        if bidx == 0:
            sample_coordinates = coordinates_batch.detach().cpu()
            sample_heatmaps = heatmaps.detach().cpu()

        if verbose and bidx < 3:
            print(f"Batch {bidx}: avg tour length = {tour_lengths.mean().item():.2f}")

    tour_np = np.asarray(all_tour_lengths, dtype=np.float64)
    results = {
        "mean_tour_length": float(np.mean(tour_np)) if tour_np.size else float("nan"),
        "std_tour_length": float(np.std(tour_np)) if tour_np.size else float("nan"),
        "min_tour_length": float(np.min(tour_np)) if tour_np.size else float("nan"),
        "max_tour_length": float(np.max(tour_np)) if tour_np.size else float("nan"),
        "median_tour_length": float(np.median(tour_np)) if tour_np.size else float("nan"),
        "total_instances": int(total_instances),
        "avg_inference_time": float(np.mean(inference_times)) if inference_times else float("nan"),
        "tour_lengths": tour_np,
        "sample_coordinates": sample_coordinates,
        "sample_heatmaps": sample_heatmaps,
    }
    return results


def compute_greedy_baseline(dataloader, verbose=True):
    """Greedy nearest-neighbor baseline (CPU/NumPy)."""
    greedy_lengths = []
    for bidx, coordinates_batch in enumerate(tqdm(dataloader, desc="Computing greedy baseline")):
        coords_b = coordinates_batch.cpu().numpy()  # (B, N, 2)
        B, N, _ = coords_b.shape
        for b in range(B):
            coords = coords_b[b]
            tour = [0]
            remaining = set(range(1, N))
            cur = 0
            while remaining:
                # find nearest unvisited
                diffs = coords[list(remaining)] - coords[cur][None, :]
                dists = np.sqrt((diffs ** 2).sum(axis=1))
                next_city = list(remaining)[int(np.argmin(dists))]
                tour.append(next_city)
                remaining.remove(next_city)
                cur = next_city
            tour.append(0)
            # length
            diffs = coords[np.array(tour[1:])] - coords[np.array(tour[:-1])]
            total_len = float(np.sqrt((diffs ** 2).sum(axis=1)).sum())
            greedy_lengths.append(total_len)
            if verbose and len(greedy_lengths) <= 3:
                print(f"Greedy tour {len(greedy_lengths)}: length = {total_len:.2f}")
    return greedy_lengths


def analyze_results(model_results, greedy_results=None, shift=-1, distance_scale=5.0):
    """Print a concise analysis summary."""
    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total instances evaluated: {model_results['total_instances']:,}")
    print(f"Average inference time: {model_results['avg_inference_time']:.4f}s per batch")
    print(f"Shift parameter used: {shift}")
    print(f"Distance scaling factor: {distance_scale}")

    print("\nTour Length Statistics:")
    print(f"  Mean   : {model_results['mean_tour_length']:.2f}")
    print(f"  Std    : {model_results['std_tour_length']:.2f}")
    print(f"  Median : {model_results['median_tour_length']:.2f}")
    print(f"  Min    : {model_results['min_tour_length']:.2f}")
    print(f"  Max    : {model_results['max_tour_length']:.2f}")

    if greedy_results is not None and len(greedy_results) > 0:
        gm = float(np.mean(greedy_results))
        gs = float(np.std(greedy_results))
        print("\n----------------------------------------")
        print("COMPARISON WITH GREEDY BASELINE")
        print("----------------------------------------")
        print(f"Greedy baseline: mean={gm:.2f}, std={gs:.2f}, n={len(greedy_results):,}")
        improvement = (gm - model_results['mean_tour_length']) / gm * 100.0
        print(f"Model vs Greedy Improvement: {improvement:.1f}%")
        print("✓ Better than greedy!" if improvement > 0 else "✗ Worse than greedy")


def save_tour_lengths(tour_lengths, save_dir, filename_prefix, shift, num_nodes,
                      model_path=None, distance_scale=5.0, method_name="SCT Model"):
    """Save per-instance tour lengths (one per line) with a metadata header."""
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, f"{filename_prefix}_shift_{shift}_size_{num_nodes}.txt")
    arr = np.asarray(tour_lengths, dtype=np.float64)

    with open(out, "w") as f:
        f.write(f"# All {method_name.lower()} tour lengths in instance order (one per line)\n")
        f.write(f"# Total instances: {len(arr)}\n")
        f.write(f"# Number of nodes: {num_nodes}\n")
        f.write(f"# Method: {method_name}\n")
        if method_name == "SCT Model":
            f.write(f"# Shift parameter: {shift}\n")
            f.write(f"# Distance scaling factor: {distance_scale}\n")
            if model_path:
                f.write(f"# Model: {model_path}\n")
        f.write(f"# Mean: {arr.mean():.6f}\n")
        f.write(f"# Std: {arr.std():.6f}\n")
        f.write(f"# Min: {arr.min():.6f}\n")
        f.write(f"# Max: {arr.max():.6f}\n")
        for v in arr:
            f.write(f"{v:.6f}\n")
    return out


# ------------------------------- main --------------------------------

def main():
    ap = argparse.ArgumentParser(description="Test trained SCT TSP model")
    ap.add_argument("--model_path", type=str, required=True, help="Path to checkpoint (.ckpt/.pt)")
    ap.add_argument("--test_data", type=str, default="data/tsp_50_uniform_test.pt")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num_nodes", type=int, default=None, help="Force number of cities (override)")
    ap.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (test-time)")
    # optional overrides
    ap.add_argument("--override_shift", type=int, default=None)
    ap.add_argument("--override_distance_scale", type=float, default=None)
    ap.add_argument("--override_noise_scale", type=float, default=None)
    # analysis
    ap.add_argument("--compute_greedy", action="store_true")
    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--num_visualize", type=int, default=8)
    ap.add_argument("--save_dir", type=str, default="test_results")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # ---------------- Load TEST data first to get actual N (cities) ----------------
    print(f"Loading test data from: {args.test_data}")
    test_coords, test_meta = load_tsp_dataset(args.test_data)
    B, actual_num_nodes, _ = test_coords.shape
    print(f"Test data: {test_coords.shape} (instances={B}, cities={actual_num_nodes})")

    # ---------------- Load model with output_dim aligned to test set ----------------
    # If user forces --num_nodes, respect it; otherwise, use test data's N
    out_dim = args.num_nodes if args.num_nodes is not None else actual_num_nodes
    model, saved_args = load_model(args.model_path, device, out_dim)

    # determine shift / distance_scale
    shift = args.override_shift if args.override_shift is not None else getattr(saved_args, "shift", -1)
    distance_scale = args.override_distance_scale if args.override_distance_scale is not None else getattr(saved_args, "distance_scale", 5.0)

    if args.override_noise_scale is not None:
        print(f"Overriding noise_scale: {getattr(saved_args, 'noise_scale', 'n/a')} -> {args.override_noise_scale}")
        setattr(saved_args, "noise_scale", float(args.override_noise_scale))

    print(f"Using shift={shift}, distance_scale={distance_scale}")

    # ---------------- Build DataLoader (single-process, torch DataLoader) ----------
    ds = SimpleTSPDataset(test_coords)
    test_loader = build_dataloader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=None,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Test batches: {len(test_loader)}")

    # ---------------- Evaluate -----------------------------------------------------
    print("\nEvaluating model...")
    results = evaluate_model(
        model,
        test_loader,
        device,
        shift=shift,
        distance_scale=distance_scale,
        enable_mc_dropout=True,
    )

    # Save per-instance lengths (allow noise-specific subdir)
    save_dir_with_noise = (
        os.path.join(args.save_dir, f"noise_{args.override_noise_scale}")
        if args.override_noise_scale is not None
        else args.save_dir
    )
    os.makedirs(save_dir_with_noise, exist_ok=True)
    out_file = save_tour_lengths(
        results["tour_lengths"],
        save_dir_with_noise,
        "all_tour_lengths",
        shift,
        actual_num_nodes,
        model_path=args.model_path,
        distance_scale=distance_scale,
        method_name="SCT Model",
    )
    print(f"Model tour lengths saved to: {out_file}")

    # Optional greedy
    greedy_results = None
    if args.compute_greedy:
        print("\nComputing greedy baseline...")
        greedy_results = compute_greedy_baseline(test_loader, verbose=True)
        print(f"Computed greedy baseline for {len(greedy_results):,} instances")

        greedy_file = save_tour_lengths(
            greedy_results,
            args.save_dir,
            "greedy_tour_lengths",
            shift,
            actual_num_nodes,
            model_path=None,
            distance_scale=distance_scale,
            method_name="Greedy Nearest Neighbor",
        )
        print(f"Greedy tour lengths saved to: {greedy_file}")

    # Summary
    analyze_results(results, greedy_results, shift=shift, distance_scale=distance_scale)

    # Optional visualization of first batch
    if args.visualize and VIZ_AVAILABLE and results["sample_coordinates"] is not None:
        print("\nCreating tour visualizations...")
        viz_dir = os.path.join(args.save_dir, f"sample_tours_shift_{shift}_scale_{distance_scale}")
        os.makedirs(viz_dir, exist_ok=True)
        coords = results["sample_coordinates"][: args.num_visualize]
        heat = results["sample_heatmaps"][: args.num_visualize]
        try:
            visualize_batch_tours(
                coords, heat, save_dir=viz_dir,
                max_plots=args.num_visualize,
                prefix=f"test_tour_shift_{shift}_scale_{distance_scale}",
            )
            print(f"Sample tour visualizations saved to: {viz_dir}")
        except Exception as e:
            print(f"[Viz] Failed: {e}")

    # Save a concise text summary
    summary_path = os.path.join(args.save_dir, f"test_results_shift_{shift}_size_{actual_num_nodes}.txt")
    with open(summary_path, "w") as f:
        f.write("SCT TSP Model Test Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Test Data: {args.test_data}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Instances: {results['total_instances']}\n")
        f.write(f"Cities: {actual_num_nodes}\n")
        f.write(f"Shift: {shift}\n")
        f.write(f"Distance scale: {distance_scale}\n\n")
        f.write("Stats (tour length):\n")
        f.write(f"  mean={results['mean_tour_length']:.4f}\n")
        f.write(f"  std={results['std_tour_length']:.4f}\n")
        f.write(f"  median={results['median_tour_length']:.4f}\n")
        f.write(f"  min={results['min_tour_length']:.4f}\n")
        f.write(f"  max={results['max_tour_length']:.4f}\n")
        f.write(f"  avg_inference_time_per_batch={results['avg_inference_time']:.4f}s\n")
        if greedy_results is not None and len(greedy_results) > 0:
            gm = float(np.mean(greedy_results))
            improvement = (gm - results['mean_tour_length']) / gm * 100.0
            f.write("\nGreedy baseline:\n")
            f.write(f"  mean={gm:.4f}\n")
            f.write(f"  improvement_vs_greedy={improvement:.2f}%\n")
    print(f"\nDetailed results saved to: {summary_path}")
    print(f"All outputs saved to: {args.save_dir}")
    print("\nTesting completed!")


if __name__ == "__main__":
    main()

