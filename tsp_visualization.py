"""
TSP Tour Visualization Functions
Visualize Hamiltonian cycles from coordinates and heatmaps
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import os

def extract_tour_from_heatmap(heatmap, threshold=0.5):
    """
    Extract tour from heatmap (permutation matrix)
    
    Args:
        heatmap: (N, N) tensor representing permutation matrix
        threshold: threshold for considering an edge as part of tour
    
    Returns:
        tour: list of city indices representing the tour
    """
    n = heatmap.shape[0]
    
    # Convert to numpy for easier processing
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()
    
    # Find the strongest connections for each city
    tour = []
    visited = set()
    
    # Start from city 0
    current_city = 0
    tour.append(current_city)
    visited.add(current_city)
    
    while len(visited) < n:
        # Find the strongest unvisited connection from current city
        connections = heatmap[current_city].copy()
        
        # Mask already visited cities
        for v in visited:
            connections[v] = -1
        
        # Find the strongest connection
        next_city = np.argmax(connections)
        
        if connections[next_city] > threshold:
            tour.append(next_city)
            visited.add(next_city)
            current_city = next_city
        else:
            # If no strong connection, find nearest unvisited
            unvisited = [i for i in range(n) if i not in visited]
            if unvisited:
                next_city = unvisited[0]  # Fallback
                tour.append(next_city)
                visited.add(next_city)
                current_city = next_city
            else:
                break
    
    # Close the loop
    if len(tour) == n:
        tour.append(tour[0])
    
    return tour

def compute_tour_length(coordinates, tour):
    """
    Compute the total length of a tour
    
    Args:
        coordinates: (N, 2) tensor of city coordinates
        tour: list of city indices
    
    Returns:
        total_length: float
    """
    if isinstance(coordinates, torch.Tensor):
        coordinates = coordinates.detach().cpu().numpy()
    
    total_length = 0
    for i in range(len(tour) - 1):
        city1 = coordinates[tour[i]]
        city2 = coordinates[tour[i + 1]]
        distance = np.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)
        total_length += distance
    
    return total_length

def visualize_tsp_tour(coordinates, heatmap, save_path=None, title=None, 
                      show_heatmap=True, show_cities=True, show_tour=True,
                      figsize=(12, 8)):
    """
    Visualize TSP tour from coordinates and heatmap
    
    Args:
        coordinates: (N, 2) tensor of city coordinates
        heatmap: (N, N) tensor representing the tour matrix (P D P^T)
        save_path: path to save the figure
        title: title for the plot
        show_heatmap: whether to show heatmap as background
        show_cities: whether to show city points
        show_tour: whether to show tour lines
        figsize: figure size
    """
    # Convert to numpy
    if isinstance(coordinates, torch.Tensor):
        coordinates = coordinates.detach().cpu().numpy()
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()
    
    n_cities = len(coordinates)
    
    # Extract tour from heatmap
    tour = extract_tour_from_heatmap(heatmap)
    tour_length = compute_tour_length(coordinates, tour)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Left plot: Heatmap
    if show_heatmap:
        im = axes[0].imshow(heatmap, cmap='Greys', interpolation='nearest')
        axes[0].set_title('Tour Heatmap (P V P^T)')
        axes[0].set_xlabel('City Index')
        axes[0].set_ylabel('City Index')
#        plt.colorbar(im, ax=axes[0])
        
#        # Add text annotations for strong connections
#        threshold = np.max(heatmap) * 0.7
#        for i in range(n_cities):
#            for j in range(n_cities):
#                if heatmap[i, j] > threshold:
#                    axes[0].text(j, i, f'{heatmap[i, j]:.2f}', 
#                               ha='center', va='center', color='white', fontsize=8)
    
    # Right plot: TSP Tour
    ax = axes[1]
    
    # Plot cities
    if show_cities:
        ax.scatter(coordinates[:, 0], coordinates[:, 1], 
                  c='red', s=100, zorder=3, alpha=0.8)
        
        # Add city labels
        for i, (x, y) in enumerate(coordinates):
            ax.annotate(str(i), (x, y), xytext=(5, 5), 
                       textcoords='offset points', fontsize=10, 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Plot tour
    if show_tour and len(tour) > 1:
        # Create line segments for the tour
        tour_coords = coordinates[tour]
        
        # Plot tour lines
        for i in range(len(tour) - 1):
            x1, y1 = tour_coords[i]
            x2, y2 = tour_coords[i + 1]
            ax.plot([x1, x2], [y1, y2], 'b-', linewidth=2, alpha=0.7)
            
            # Add arrow to show direction
            dx, dy = x2 - x1, y2 - y1
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
    
    # Set title and labels
    if title is None:
        title = f'TSP Tour (Length: {tour_length:.2f})'
    ax.set_title(title)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add tour information
    tour_text = f'Tour: {" → ".join(map(str, tour[:10]))}{"..." if len(tour) > 10 else ""}'
    ax.text(0.02, 0.98, tour_text, transform=ax.transAxes, 
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
           verticalalignment='top', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Tour visualization saved to: {save_path}")
    
    plt.show()
    
    return tour, tour_length

def visualize_batch_tours(coordinates_batch, heatmaps_batch, save_dir='tour_visualizations', 
                         max_plots=4, prefix='tour'):
    """
    Visualize multiple TSP tours from a batch
    
    Args:
        coordinates_batch: (B, N, 2) batch of coordinates
        heatmaps_batch: (B, N, N) batch of heatmaps
        save_dir: directory to save visualizations
        max_plots: maximum number of tours to visualize
        prefix: prefix for saved files
    """
    batch_size = coordinates_batch.shape[0]
    num_plots = min(batch_size, max_plots)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    tours = []
    lengths = []
    
    for i in range(num_plots):
        coordinates = coordinates_batch[i]  # (N, 2)
        heatmap = heatmaps_batch[i]  # (N, N)
        
        save_path = os.path.join(save_dir, f'{prefix}_instance_{i}.png')
        title = f'TSP Instance {i}'
        
        tour, tour_length = visualize_tsp_tour(
            coordinates, heatmap, 
            save_path=save_path, 
            title=title,
            figsize=(10, 6)
        )
        
        tours.append(tour)
        lengths.append(tour_length)
        
        plt.close()  # Close to avoid memory issues
    
    # Summary
    print(f"\nVisualized {num_plots} tours:")
    for i, length in enumerate(lengths):
        print(f"  Instance {i}: Tour length = {length:.2f}")
    print(f"  Average tour length: {np.mean(lengths):.2f}")
    print(f"  Visualizations saved to: {save_dir}")
    
    return tours, lengths

def visualize_validation_tours(coordinates_batch, val_heatmaps, epoch, save_dir='validation_tours'):
    """
    Visualize validation tours during training
    
    Args:
        coordinates_batch: (B, N, 2) batch of coordinates  
        val_heatmaps: (B, N, N) validation heatmaps
        epoch: current epoch number
        save_dir: directory to save visualizations
    """
    epoch_dir = os.path.join(save_dir, f'epoch_{epoch:03d}')
    
    tours, lengths = visualize_batch_tours(
        coordinates_batch, val_heatmaps,
        save_dir=epoch_dir,
        max_plots=4,
        prefix=f'val_epoch_{epoch}'
    )
    
    # Save summary
    summary_path = os.path.join(epoch_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Validation Results - Epoch {epoch}\n")
        f.write("=" * 40 + "\n")
        f.write(f"Number of instances: {len(lengths)}\n")
        f.write(f"Average tour length: {np.mean(lengths):.2f}\n")
        f.write(f"Best tour length: {np.min(lengths):.2f}\n")
        f.write(f"Worst tour length: {np.max(lengths):.2f}\n")
        f.write(f"Standard deviation: {np.std(lengths):.2f}\n")
        f.write("\nIndividual tour lengths:\n")
        for i, length in enumerate(lengths):
            f.write(f"  Instance {i}: {length:.2f}\n")
    
    return tours, lengths

# Example usage function
def test_visualization():
    """Test the visualization functions"""
    print("Testing TSP visualization...")
    
    # Create test data
    torch.manual_seed(42)
    n_cities = 10
    coordinates = torch.rand(n_cities, 2) * 100
    
    # Create a fake heatmap (random permutation matrix)
    heatmap = torch.rand(n_cities, n_cities)
    heatmap = heatmap + heatmap.t()  # Make symmetric
    heatmap.fill_diagonal_(0)  # No self-loops
    
    # Visualize
    tour, length = visualize_tsp_tour(
        coordinates, heatmap,
        save_path='test_tour.png',
        title='Test TSP Tour'
    )
    
    print(f"Test tour: {tour}")
    print(f"Test tour length: {length:.2f}")
    print("✓ Visualization test completed!")

if __name__ == "__main__":
    test_visualization()
