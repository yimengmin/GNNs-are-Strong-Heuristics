"""
Test script for trained SCT TSP model
Evaluate the model on test data and provide comprehensive analysis
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import time
from tqdm import tqdm

# Import modules

#from models import GNN
#from cyclefastsags import EnhancedUltraFastSCTGNN as GNN
from cyclefastsags_stable import EnhancedUltraFastSCTGNN as GNN

from data_generator import SimpleTSPDataset, SimpleTSPDataLoader, load_tsp_dataset
from hardpermutation import to_exact_permutation_batched
from tsp_visualization import visualize_batch_tours, compute_tour_length, extract_tour_from_heatmap
from utsploss import tsp_permutation_loss
def load_model(model_path, device, num_nodes=None):
    """Load trained model from checkpoint"""
    print(f"Loading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    args = checkpoint['args']
    
    print(f"Model trained for {checkpoint['epoch']+1} epochs")
    print(f"Best validation length: {checkpoint['val_length']:.2f}")
    print(f"Shift parameter: {getattr(args, 'shift', -1)}")
    print(f"Distance scaling factor: {getattr(args, 'distance_scale', 5.0)}")
    print(f"Number of layers: {getattr(args, 'n_layers', 2)}")
    
    # Determine output dimension (number of nodes)
    if num_nodes is not None:
        output_dim = num_nodes
        print(f"Using provided number of nodes: {output_dim}")
    else:
        # Try to get from saved args, fallback to 50
        output_dim = getattr(args, 'num_nodes', 50)
        print(f"Using number of nodes from model args: {output_dim}")
    
    # Create model with same architecture
    model = GNN(
        input_dim=2,  # x, y coordinates
        hidden_dim=args.hidden_dim,
        output_dim=output_dim,
        n_layers=args.n_layers,
        sct_order=args.sct_order,
        gcn_order=args.gcn_order,
        tanh_scale=args.tanh_scale,
        tau=args.tau,
        n_iter=args.n_iter,
        noise_scale=args.noise_scale,
        inference_mode=True  # Set to inference mode
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model output dimension: {output_dim}")
    
    return model, args
def enable_dropout_only(model):
    """
    Turn on dropout during inference by setting only Dropout layers to training mode.
    Leaves the rest of the model (including BatchNorm) in eval mode.
    """
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()

def evaluate_model(model, dataloader, device, shift=-1, distance_scale=5.0, verbose=True):
    """
    Evaluate model on test data
    
    Returns:
        results: dict containing evaluation metrics
    """
    model.eval()
    enable_dropout_only(model)    #enable Dropout


    
    all_tour_lengths = []
    all_coordinates = []
    all_heatmaps = []
    inference_times = []
    
    total_instances = 0
    
    with torch.no_grad():
        for batch_idx, coordinates_batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            coordinates_batch = coordinates_batch.to(device)
            batch_size, n_cities = coordinates_batch.shape[:2]
            
            # Compute distance matrices
            distance_matrices = torch.cdist(coordinates_batch, coordinates_batch)
            
            # Create adjacency matrices with distance scaling factor
            adjacency_matrices = torch.exp(-1 * distance_matrices / distance_scale)
            
            # Node features are coordinates
            node_features = coordinates_batch
            
            # Time the inference
            start_time = time.time()
            
            # Get model output (logits in inference mode)
            logits = model(node_features, adjacency_matrices)
            
            # Convert to hard permutation matrices
            P_hard = to_exact_permutation_batched(logits)
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Compute tour lengths with the same shift parameter used in training
            tour_lengths, heatmaps = tsp_permutation_loss(P_hard, distance_matrices, shift)
            
            # Store results
            all_tour_lengths.extend(tour_lengths.cpu().numpy())
            
            # Store first batch for visualization
            if batch_idx == 0:
                all_coordinates = coordinates_batch.cpu()
                all_heatmaps = heatmaps.cpu()
            
            total_instances += batch_size
            
            if verbose and batch_idx < 3:  # Print first few batches
                print(f"Batch {batch_idx}: avg tour length = {tour_lengths.mean():.2f}")
    
    # Compute statistics
    tour_lengths_np = np.array(all_tour_lengths)
    results = {
        'mean_tour_length': np.mean(tour_lengths_np),
        'std_tour_length': np.std(tour_lengths_np),
        'min_tour_length': np.min(tour_lengths_np),
        'max_tour_length': np.max(tour_lengths_np),
        'median_tour_length': np.median(tour_lengths_np),
        'total_instances': total_instances,
        'avg_inference_time': np.mean(inference_times),
        'tour_lengths': tour_lengths_np,
        'sample_coordinates': all_coordinates,
        'sample_heatmaps': all_heatmaps
    }
    
    return results

def compute_greedy_baseline(dataloader, verbose=True):
    """
    Compute greedy nearest neighbor baseline for comparison on entire dataset
    
    Args:
        dataloader: DataLoader containing coordinates
        
    Returns:
        greedy_lengths: list of tour lengths for all instances
    """
    greedy_lengths = []
    
    for batch_idx, coordinates_batch in enumerate(tqdm(dataloader, desc="Computing greedy baseline")):
        batch_size, n_cities = coordinates_batch.shape[:2]
        
        for b in range(batch_size):
            coords = coordinates_batch[b].numpy()
            
            # Greedy nearest neighbor starting from city 0
            tour = [0]
            remaining = set(range(1, n_cities))
            current = 0
            
            while remaining:
                # Find nearest unvisited city
                distances = []
                for city in remaining:
                    dist = np.sqrt(np.sum((coords[current] - coords[city])**2))
                    distances.append((dist, city))
                
                # Go to nearest city
                _, next_city = min(distances)
                tour.append(next_city)
                remaining.remove(next_city)
                current = next_city
            
            # Return to start
            tour.append(0)
            
            # Compute tour length
            total_length = 0
            for i in range(len(tour) - 1):
                dist = np.sqrt(np.sum((coords[tour[i]] - coords[tour[i+1]])**2))
                total_length += dist
            
            greedy_lengths.append(total_length)
            
            if verbose and len(greedy_lengths) <= 3:
                print(f"Greedy tour {len(greedy_lengths)}: length = {total_length:.2f}")
    
    return greedy_lengths

def analyze_results(model_results, greedy_results=None, shift=-1, distance_scale=5.0):
    """Analyze and compare results"""
    
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    
    print(f"Total instances evaluated: {model_results['total_instances']:,}")
    print(f"Average inference time: {model_results['avg_inference_time']:.4f}s per batch")
    print(f"Shift parameter used: {shift}")
    print(f"Distance scaling factor: {distance_scale}")
    
    print(f"\nTour Length Statistics:")
    print(f"  Mean:   {model_results['mean_tour_length']:.2f}")
    print(f"  Std:    {model_results['std_tour_length']:.2f}")
    print(f"  Median: {model_results['median_tour_length']:.2f}")
    print(f"  Min:    {model_results['min_tour_length']:.2f}")
    print(f"  Max:    {model_results['max_tour_length']:.2f}")
    
    # Percentiles
    tour_lengths = model_results['tour_lengths']
    percentiles = [10, 25, 75, 90, 95, 99]
    print(f"\nPercentiles:")
    for p in percentiles:
        value = np.percentile(tour_lengths, p)
        print(f"  {p:2d}th: {value:.2f}")
    
    # Comparison with greedy baseline
    if greedy_results is not None:
        greedy_mean = np.mean(greedy_results)
        greedy_std = np.std(greedy_results)
        
        print(f"\n" + "-"*40)
        print("COMPARISON WITH GREEDY BASELINE")
        print("-"*40)
        print(f"Greedy baseline:")
        print(f"  Mean: {greedy_mean:.2f}")
        print(f"  Std:  {greedy_std:.2f}")
        print(f"  Instances: {len(greedy_results):,}")
        
        improvement = (greedy_mean - model_results['mean_tour_length']) / greedy_mean * 100
        print(f"\nModel vs Greedy:")
        print(f"  Improvement: {improvement:.1f}%")
        
        if improvement > 0:
            print(f"  ✓ Model is {improvement:.1f}% better than greedy!")
        else:
            print(f"  ✗ Model is {abs(improvement):.1f}% worse than greedy")
        
        # Statistical significance test
        try:
            from scipy import stats
            if len(greedy_results) == len(tour_lengths):
                t_stat, p_value = stats.ttest_rel(greedy_results, tour_lengths)
                print(f"  T-test p-value: {p_value:.2e}")
                if p_value < 0.05:
                    print(f"  ✓ Difference is statistically significant (p < 0.05)")
                else:
                    print(f"  ✗ Difference is not statistically significant")
            else:
                print(f"  ⚠ Cannot perform paired t-test: different sample sizes")
                print(f"    Model: {len(tour_lengths):,} instances")
                print(f"    Greedy: {len(greedy_results):,} instances")
        except ImportError:
            print("  scipy not available for statistical test")

def create_performance_plots(model_results, greedy_results=None, save_dir='test_results', 
                           shift=-1, distance_scale=5.0,graph_size=50):
    """Create performance analysis plots"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    tour_lengths = model_results['tour_lengths']
    
    # Distribution plot
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Histogram
    plt.subplot(2, 2, 1)
    plt.hist(tour_lengths, bins=50, alpha=0.7, color='blue', 
             label=f'SCT Model (shift={shift}, size={graph_size})')
    if greedy_results is not None:
        plt.hist(greedy_results, bins=50, alpha=0.7, color='red', label='Greedy Baseline')
    plt.xlabel('Tour Length')
    plt.ylabel('Frequency')
    plt.title('Distribution of Tour Lengths')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Box plot
    plt.subplot(2, 2, 2)
    data_to_plot = [tour_lengths]
    labels = [f'SCT (shift={shift}, size={graph_size})']
    if greedy_results is not None:
        data_to_plot.append(greedy_results)
        labels.append('Greedy')
    
    plt.boxplot(data_to_plot, labels=labels)
    plt.ylabel('Tour Length')
    plt.title('Tour Length Comparison')
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Cumulative distribution
    plt.subplot(2, 2, 3)
    sorted_lengths = np.sort(tour_lengths)
    percentiles = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths) * 100
    plt.plot(sorted_lengths, percentiles, 
             label=f'SCT Model (shift={shift}, size={graph_size})', linewidth=2)
    
    if greedy_results is not None:
        sorted_greedy = np.sort(greedy_results)
        greedy_percentiles = np.arange(1, len(sorted_greedy) + 1) / len(sorted_greedy) * 100
        plt.plot(sorted_greedy, greedy_percentiles, label='Greedy', linewidth=2)
    
    plt.xlabel('Tour Length')
    plt.ylabel('Cumulative Percentage')
    plt.title('Cumulative Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Performance scatter (if we have greedy baseline)
    plt.subplot(2, 2, 4)
    if greedy_results is not None and len(greedy_results) == len(tour_lengths):
        plt.scatter(greedy_results, tour_lengths, alpha=0.6, s=20)
        
        # Perfect performance line
        min_val = min(min(greedy_results), min(tour_lengths))
        max_val = max(max(greedy_results), max(tour_lengths))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Equal Performance')
        
        plt.xlabel('Greedy Tour Length')
        plt.ylabel('SCT Model Tour Length')
        plt.title('Instance-wise Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        # Just show tour lengths over instances
        plt.plot(tour_lengths, alpha=0.7)
        plt.xlabel('Instance')
        plt.ylabel('Tour Length')
        plt.title('Tour Lengths by Instance')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_dir, 
                            f'performance_analysis_shift_{shift}_size_{graph_size}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Performance plots saved to: {plot_path}")


def save_tour_lengths(tour_lengths, save_dir, filename_prefix, shift, num_nodes,
                     model_path=None, distance_scale=5.0, method_name="SCT Model"):
    """Save tour lengths to file with metadata"""
    tour_lengths_file = os.path.join(save_dir, f'{filename_prefix}_shift_{shift}_size_{num_nodes}.txt')

    with open(tour_lengths_file, 'w') as f:
        # Write header information
        f.write(f"# All {method_name.lower()} tour lengths in instance order (one per line)\n")
        f.write(f"# Total instances: {len(tour_lengths)}\n")
        f.write(f"# Number of nodes: {num_nodes}\n")
        f.write(f"# Method: {method_name}\n")

        if method_name == "SCT Model":
            f.write(f"# Shift parameter: {shift}\n")
            f.write(f"# Distance scaling factor: {distance_scale}\n")
            if model_path:
                f.write(f"# Model: {model_path}\n")

        f.write(f"# Mean: {np.mean(tour_lengths):.6f}\n")
        f.write(f"# Std: {np.std(tour_lengths):.6f}\n")
        f.write(f"# Min: {np.min(tour_lengths):.6f}\n")
        f.write(f"# Max: {np.max(tour_lengths):.6f}\n")

        # Each line one tour length
        for length in tour_lengths:
            f.write(f"{length:.6f}\n")

    return tour_lengths_file


def main():
    parser = argparse.ArgumentParser(description='Test trained SCT TSP model')
    
    # Model and data parameters
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--test_data', type=str, default='data/tsp_50_uniform_test.pt',
                       help='Path to test dataset')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for testing')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on')
    parser.add_argument('--num_nodes', type=int, default=None,
                       help='Number of nodes (cities) in TSP instances. If not specified, will use value from model checkpoint')

    parser.add_argument('--override_noise_scale', type=float, default=None,
            help='Override noise scale factor (use model default if None--- not specified)')


    # Override parameters (optional - will use model's saved parameters by default)
    parser.add_argument('--override_shift', type=int, default=None,
                       help='Override shift parameter (use model default if not specified)')
    parser.add_argument('--override_distance_scale', type=float, default=None,
                       help='Override distance scaling factor (use model default if not specified)')
    
    # Analysis options
    parser.add_argument('--compute_greedy', action='store_true',
                       help='Compute greedy baseline for comparison')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations of sample tours')
    parser.add_argument('--save_dir', type=str, default='test_results',
                       help='Directory to save results')
    parser.add_argument('--num_visualize', type=int, default=8,
                       help='Number of tours to visualize')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create results directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load model
    model, model_args = load_model(args.model_path, device, args.num_nodes)

    if args.override_noise_scale is not None:
        print(f"Overriding noise_scale: {model_args.noise_scale} -> {args.override_noise_scale}")
        model_args.noise_scale = args.override_noise_scale

    # Get parameters from model args (with optional overrides)
    shift = args.override_shift if args.override_shift is not None else getattr(model_args, 'shift', -1)
    distance_scale = (args.override_distance_scale if args.override_distance_scale is not None 
                     else getattr(model_args, 'distance_scale', 5.0))
    
    print(f"Using shift parameter: {shift}")
    print(f"Using distance scaling factor: {distance_scale}")
    
    if args.override_shift is not None:
        print(f"  (Overridden from model default: {getattr(model_args, 'shift', -1)})")
    if args.override_distance_scale is not None:
        print(f"  (Overridden from model default: {getattr(model_args, 'distance_scale', 5.0)})")
    
    # Load test data
    print(f"Loading test data from: {args.test_data}")
    test_coords, test_metadata = load_tsp_dataset(args.test_data)
    
    print(f"Test data: {test_coords.shape}")
    print(f"Distribution: {test_metadata.get('distribution', 'unknown')}")
    
    # Validate that the number of nodes matches the test data
    _, actual_num_nodes, _ = test_coords.shape
    expected_num_nodes = args.num_nodes if args.num_nodes is not None else getattr(model_args, 'num_nodes', 50)
    
    if actual_num_nodes != expected_num_nodes:
        print(f"WARNING: Test data has {actual_num_nodes} nodes, but model expects {expected_num_nodes} nodes!")
        print(f"This may cause dimension mismatch errors.")
    
    # Create dataset and dataloader
    test_dataset = SimpleTSPDataset(test_coords)
    test_loader = SimpleTSPDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Test batches: {len(test_loader)}")
    
    # Evaluate model
    print("\nEvaluating model...")
    model_results = evaluate_model(model, test_loader, device, 
                                  shift=shift, distance_scale=distance_scale)

    # Save model tour lengths
    save_dir_with_noise = (
        os.path.join(args.save_dir, f"noise_{args.override_noise_scale}")
        if args.override_noise_scale is not None
        else args.save_dir
    )
    os.makedirs(save_dir_with_noise, exist_ok=True) 
    model_tour_lengths_file = save_tour_lengths(
        model_results['tour_lengths'], 
        #args.save_dir, 
        save_dir_with_noise,
        'all_tour_lengths',
        #'Faster_HPI_all_tour_lengths', 
        shift, 
        actual_num_nodes,
        model_path=args.model_path,
        distance_scale=distance_scale,
        method_name="SCT Model"
    )
    print(f"Model tour lengths saved to: {model_tour_lengths_file}")

    # Compute greedy baseline if requested
    greedy_results = None
    if args.compute_greedy:
        print("\nComputing greedy baseline...")
        # FIXED: Use the same dataloader to compute greedy for ALL instances
        greedy_results = compute_greedy_baseline(test_loader, verbose=True)
        
        print(f"Computed greedy baseline for {len(greedy_results):,} instances")
        print(f"Model evaluated on {len(model_results['tour_lengths']):,} instances")
        
        # Verify we have the same number of instances
        if len(greedy_results) != len(model_results['tour_lengths']):
            print(f"WARNING: Mismatch in number of instances!")
            print(f"  Model: {len(model_results['tour_lengths']):,}")
            print(f"  Greedy: {len(greedy_results):,}")
        else:
            print(f"✓ Same number of instances for fair comparison")

        # Save greedy tour lengths
        greedy_tour_lengths_file = save_tour_lengths(
            greedy_results,
            args.save_dir,
            'greedy_tour_lengths',
            shift,  # Keep shift in filename for consistency, though greedy doesn't use it
            actual_num_nodes,
            method_name="Greedy Nearest Neighbor"
        )
        print(f"Greedy tour lengths saved to: {greedy_tour_lengths_file}")
    
    # Analyze results
    analyze_results(model_results, greedy_results, shift=shift, distance_scale=distance_scale)
    
    # Create performance plots
#    create_performance_plots(model_results, greedy_results, args.save_dir, 
#                           shift=shift, distance_scale=distance_scale,graph_size=actual_num_nodes)
    
    # Visualize sample tours
    if args.visualize and model_results['sample_coordinates'] is not None:
        print(f"\nCreating tour visualizations...")
        viz_dir = os.path.join(args.save_dir, f'sample_tours_shift_{shift}_scale_{distance_scale}')
        
        sample_coords = model_results['sample_coordinates'][:args.num_visualize]
        sample_heatmaps = model_results['sample_heatmaps'][:args.num_visualize]
        
        tours, lengths = visualize_batch_tours(
            sample_coords, sample_heatmaps,
            save_dir=viz_dir,
            max_plots=args.num_visualize,
            prefix=f'test_tour_shift_{shift}_scale_{distance_scale}'
        )
        
        print(f"Sample tour visualizations saved to: {viz_dir}")
    
    # Save detailed results
    results_file = os.path.join(args.save_dir, 
                               f'test_results_shift_{shift}_size_{actual_num_nodes}.txt')
    with open(results_file, 'w') as f:
        f.write("SCT TSP Model Test Results\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Test Data: {args.test_data}\n")
        f.write(f"Distribution: {test_metadata.get('distribution', 'unknown')}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Number of nodes: {expected_num_nodes}\n")
        f.write(f"Shift parameter: {shift}\n")
        f.write(f"Distance scaling factor: {distance_scale}\n\n")
        
        f.write(f"Model Architecture:\n")
        f.write(f"  Hidden dim: {model_args.hidden_dim}\n")
        f.write(f"  Output dim: {expected_num_nodes}\n")
        f.write(f"  Layers: {model_args.n_layers}\n")
        f.write(f"  SCT order: {model_args.sct_order}\n")
        f.write(f"  GCN order: {model_args.gcn_order}\n")
        f.write(f"  Tau: {model_args.tau}\n")
        f.write(f"  N_iter: {model_args.n_iter}\n")
        f.write(f"  Noise scale: {model_args.noise_scale}\n")
        f.write(f"  Shift: {shift}\n")
        f.write(f"  Distance scale: {distance_scale}\n\n")
        
        f.write(f"Test Results:\n")
        f.write(f"  Total instances: {model_results['total_instances']:,}\n")
        f.write(f"  Mean tour length: {model_results['mean_tour_length']:.4f}\n")
        f.write(f"  Std tour length: {model_results['std_tour_length']:.4f}\n")
        f.write(f"  Median tour length: {model_results['median_tour_length']:.4f}\n")
        f.write(f"  Min tour length: {model_results['min_tour_length']:.4f}\n")
        f.write(f"  Max tour length: {model_results['max_tour_length']:.4f}\n")
        f.write(f"  Avg inference time: {model_results['avg_inference_time']:.4f}s\n")
        
        if greedy_results:
            f.write(f"\nGreedy Baseline:\n")
            f.write(f"  Total instances: {len(greedy_results):,}\n")
            f.write(f"  Mean tour length: {np.mean(greedy_results):.4f}\n")
            f.write(f"  Std tour length: {np.std(greedy_results):.4f}\n")
            improvement = ((np.mean(greedy_results) - model_results['mean_tour_length']) / 
                          np.mean(greedy_results) * 100)
            f.write(f"  Improvement: {improvement:.2f}%\n")
    
    print(f"\nDetailed results saved to: {results_file}")
    print(f"All outputs saved to: {args.save_dir}")
    print("\nTesting completed!")

if __name__ == "__main__":
    main()
