"""
Beam Search Implementation for TSP: https://github.com/danilonumeroso/conar/tree/main/baselines
"""

import torch
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from statistics import mean, stdev as std
import time

@partial(jax.jit, static_argnames=['beam_width', 'n_cities'])
def beam_search_tsp_single(distance_matrix, beam_width, n_cities, start_city=0):
    """
    Beam search for single TSP instance
    
    Args:
        distance_matrix: (n_cities, n_cities) distance matrix
        beam_width: beam width
        n_cities: number of cities
        start_city: starting city (default 0)
    
    Returns:
        best_tour: array of city indices representing the tour
        best_cost: cost of the best tour
    """
    
    # Initialize beam: each beam state is (visited_mask, current_city, cost)
    # visited_mask: boolean array indicating which cities are visited
    # current_city: current city in the tour
    # cost: total cost so far
    
    initial_visited = jnp.zeros(n_cities, dtype=bool).at[start_city].set(True)
    beam_visited = initial_visited[None, :]  # (1, n_cities)
    beam_current = jnp.array([start_city])   # (1,)
    beam_cost = jnp.array([0.0])             # (1,)
    beam_paths = jnp.array([[start_city] + [-1] * (n_cities - 1)])  # (1, n_cities)
    
    # For each step, expand beam and keep best candidates
    for step in range(1, n_cities):
        # Current beam size
        current_beam_size = beam_visited.shape[0]
        
        # Generate all possible next cities for each beam state
        # For each beam state, we can go to any unvisited city
        next_cities = jnp.arange(n_cities)  # All possible next cities
        
        # Expand: create all (beam_state, next_city) combinations
        # Shape expansions: (current_beam_size, n_cities, ...)
        expanded_visited = jnp.repeat(beam_visited[:, None, :], n_cities, axis=1)  # (beam_size, n_cities, n_cities)
        expanded_current = jnp.repeat(beam_current[:, None], n_cities, axis=1)      # (beam_size, n_cities)
        expanded_cost = jnp.repeat(beam_cost[:, None], n_cities, axis=1)            # (beam_size, n_cities)
        expanded_paths = jnp.repeat(beam_paths[:, None, :], n_cities, axis=1)       # (beam_size, n_cities, n_cities)
        
        # For each expansion, update state
        next_cities_expanded = jnp.repeat(next_cities[None, :], current_beam_size, axis=0)  # (beam_size, n_cities)
        
        # Check which transitions are valid (to unvisited cities)
        is_valid = ~expanded_visited[jnp.arange(current_beam_size)[:, None], next_cities_expanded, next_cities_expanded]
        
        # Update visited mask
        new_visited = expanded_visited.at[
            jnp.arange(current_beam_size)[:, None], 
            jnp.arange(n_cities)[None, :], 
            next_cities_expanded
        ].set(True)
        
        # Update cost
        transition_costs = distance_matrix[expanded_current, next_cities_expanded]
        new_cost = expanded_cost + transition_costs
        
        # Update paths
        new_paths = expanded_paths.at[
            jnp.arange(current_beam_size)[:, None],
            jnp.arange(n_cities)[None, :],
            step
        ].set(next_cities_expanded)
        
        # Flatten all expansions
        new_visited_flat = new_visited.reshape(-1, n_cities)
        new_current_flat = next_cities_expanded.reshape(-1)
        new_cost_flat = new_cost.reshape(-1)
        new_paths_flat = new_paths.reshape(-1, n_cities)
        is_valid_flat = is_valid.reshape(-1)
        
        # Filter out invalid transitions
        valid_indices = jnp.where(is_valid_flat)[0]
        
        if len(valid_indices) == 0:
            break
            
        valid_visited = new_visited_flat[valid_indices]
        valid_current = new_current_flat[valid_indices]
        valid_cost = new_cost_flat[valid_indices]
        valid_paths = new_paths_flat[valid_indices]
        
        # Select top beam_width candidates
        num_candidates = min(beam_width, len(valid_indices))
        top_indices = jnp.argsort(valid_cost)[:num_candidates]
        
        beam_visited = valid_visited[top_indices]
        beam_current = valid_current[top_indices]
        beam_cost = valid_cost[top_indices]
        beam_paths = valid_paths[top_indices]
    
    # Add return to start city
    return_costs = distance_matrix[beam_current, start_city]
    final_costs = beam_cost + return_costs
    
    # Select best solution
    best_idx = jnp.argmin(final_costs)
    best_cost = final_costs[best_idx]
    best_path = beam_paths[best_idx]
    
    return best_path, best_cost

# Simplified version that's easier to understand and debug
def beam_search_tsp_simple(distance_matrix, beam_width, start_city=0):
    """
    Simple, non-JIT beam search for TSP (easier to debug)
    """
    n_cities = distance_matrix.shape[0]
    
    # Each beam state: (visited_set, current_city, cost, path)
    beam = [(frozenset([start_city]), start_city, 0.0, [start_city])]
    
    for step in range(1, n_cities):
        new_beam = []
        
        for visited, current_city, cost, path in beam:
            # Find unvisited cities
            unvisited = set(range(n_cities)) - visited
            
            # Expand to each unvisited city
            for next_city in unvisited:
                new_visited = visited | {next_city}
                new_cost = cost + distance_matrix[current_city, next_city]
                new_path = path + [next_city]
                new_beam.append((new_visited, next_city, new_cost, new_path))
        
        # Sort by cost and keep best beam_width states
        new_beam.sort(key=lambda x: x[2])
        beam = new_beam[:beam_width]
    
    # Add return to start
    final_solutions = []
    for visited, current_city, cost, path in beam:
        final_cost = cost + distance_matrix[current_city, start_city]
        final_path = path
        final_solutions.append((final_cost, final_path))
    
    final_solutions.sort(key=lambda x: x[0])
    best_cost, best_path = final_solutions[0]
    
    return np.array(best_path), best_cost

def beam_search_tsp_batch_simple(distance_matrices, beam_width):
    """
    Batch beam search using simple implementation
    """
    results = []
    for i, dm in enumerate(distance_matrices):
        path, cost = beam_search_tsp_simple(dm, beam_width)
        results.append((path, cost))
    return results

# Vectorized implementation (more complex but faster)
@jax.jit
def beam_search_tsp_vectorized(distance_matrices, beam_width):
    """
    Vectorized beam search for batch processing
    """
    batch_size, n_cities, _ = distance_matrices.shape
    
    def single_instance(dm):
        return beam_search_tsp_single(dm, beam_width, n_cities, start_city=0)
    
    # Vectorize over batch dimension
    vmapped_search = jax.vmap(single_instance)
    return vmapped_search(distance_matrices)

def coords_to_distance_matrix(coordinates):
    """Convert coordinates to distance matrix"""
    if coordinates.dim() == 2:
        coordinates = coordinates.unsqueeze(0)
    distance_matrix = torch.cdist(coordinates, coordinates)
    return distance_matrix.squeeze(0) if distance_matrix.shape[0] == 1 else distance_matrix

def evaluate_corrected_beam_search(dataset_path, beam_widths=[32, 64, 128, 256], max_instances=100):
    """
    Evaluate the corrected beam search implementation
    """
    print(f"Loading TSP dataset from: {dataset_path}")
    
    # Load the dataset
    data = torch.load(dataset_path)
    coordinates = data['coordinates']
    num_instances = coordinates.shape[0]
    num_cities = coordinates.shape[1]
    
    if max_instances is not None:
        num_instances = min(num_instances, max_instances)
        coordinates = coordinates[:num_instances]
    
    print(f"Dataset: {num_instances} instances, {num_cities} cities each")
    
    # Pre-process distance matrices
    print("Computing distance matrices...")
    distance_matrices = []
    for i in range(num_instances):
        dm = coords_to_distance_matrix(coordinates[i]).numpy()
        distance_matrices.append(dm)
    
    results = {}
    
    for beam_width in beam_widths:
        print(f"\n{'='*50}")
        print(f"Testing Beam Width: {beam_width}")
        print(f"{'='*50}")
        
        start_time = time.time()
        
        # Use simple implementation for clarity and debugging
        all_costs = []
        for i, dm in enumerate(distance_matrices):
            if i % 20 == 0:
                print(f"  Processing instance {i+1}/{num_instances}")
            
            path, cost = beam_search_tsp_simple(dm, beam_width)
            all_costs.append(cost)
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        avg_cost = mean(all_costs)
        std_cost = std(all_costs) if len(all_costs) > 1 else 0
        min_cost = min(all_costs)
        max_cost = max(all_costs)
        
        results[beam_width] = {
            'average_cost': avg_cost,
            'std_cost': std_cost,
            'min_cost': min_cost,
            'max_cost': max_cost,
            'total_time': total_time,
            'avg_time_per_instance': total_time / num_instances
        }
        
        print(f"Average cost: {avg_cost:.4f}")
        print(f"Time per instance: {total_time/num_instances:.4f}s")
    
    return results

def print_comparison_results(results):
    """Print comparison results"""
    print("\n" + "="*80)
    print("CORRECTED BEAM SEARCH TSP PERFORMANCE COMPARISON")
    print("="*80)
    
    print(f"{'Beam Width':<12} {'Avg Cost':<12} {'Std Dev':<12} {'Min Cost':<12} {'Max Cost':<12} {'Time/Inst':<12}")
    print("-" * 80)
    
    beam_widths = sorted(results.keys())
    for bw in beam_widths:
        r = results[bw]
        print(f"{bw:<12} {r['average_cost']:<12.4f} {r['std_cost']:<12.4f} "
              f"{r['min_cost']:<12.4f} {r['max_cost']:<12.4f} {r['avg_time_per_instance']:<12.4f}")
    
    print("="*80)
    
    # Analysis
    costs = [results[bw]['average_cost'] for bw in beam_widths]
    times = [results[bw]['avg_time_per_instance'] for bw in beam_widths]
    
    print(f"\nAnalysis:")
    best_cost = min(costs)
    best_bw = beam_widths[costs.index(best_cost)]
    improvement = (costs[0] - best_cost) / costs[0] * 100
    
    print(f"Best average cost: {best_cost:.4f} (beam width {best_bw})")
    print(f"Improvement from smallest to best beam: {improvement:.2f}%")
    print(f"Cost range: {min(costs):.4f} - {max(costs):.4f}")
    print(f"Time range: {min(times):.4f}s - {max(times):.4f}s")
    
    # Check for consistent improvement
    improvements = []
    for i in range(len(costs) - 1):
        imp = (costs[i] - costs[i+1]) / costs[i] * 100
        improvements.append(imp)
    
    print(f"Step-wise improvements: {[f'{imp:.3f}%' for imp in improvements]}")
    
    if all(imp >= 0 for imp in improvements):
        print("Beam search is working correctly - consistent improvement with larger beams!")
    else:
        print("Mixed results - some beam widths perform worse than smaller ones")

def test_small_example():
    """Test on a very small example to verify correctness"""
    print("="*60)
    print("TESTING ON SMALL EXAMPLE")
    print("="*60)
    
    # Create a simple 5-city example
    np.random.seed(42)
    coords = np.random.rand(5, 2) * 10
    dm = np.sqrt(np.sum((coords[:, None] - coords[None, :]) ** 2, axis=-1))
    
    print("5-city distance matrix:")
    print(dm.round(2))
    print()
    
    # Test different beam widths
    for bw in [1, 2, 4, 8, 16]:
        path, cost = beam_search_tsp_simple(dm, bw)
        print(f"Beam width {bw:2d}: cost = {cost:.3f}, path = {path}")
    
    print("\nThis should show clear improvement with larger beam widths!")
    print("="*60)

if __name__ == "__main__":
    # Test small example first
    test_small_example()
    
    # Run on actual dataset
    dataset_path = '../data/tsp_20_uniform_test.pt'
    
    results = evaluate_corrected_beam_search(
        dataset_path=dataset_path,
        #beam_widths=[128,512,1280],
        beam_widths=[5000,10000],
        max_instances=1000  # Start with smaller number for testing
    )
    
    print_comparison_results(results)
