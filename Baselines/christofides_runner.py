import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_networkx
from statistics import mean, stdev as std
import os

def coords_to_complete_graph_data(coordinates):
    """
    Convert coordinates to PyTorch Geometric Data object with complete graph
    
    Args:
        coordinates: (N, 2) coordinate tensor for single instance
    
    Returns:
        Data object with complete graph structure
    """
    num_nodes = coordinates.shape[0]
    
    # Create complete graph edges
    edge_index = []
    edge_attr = []
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edge_index.append([i, j])
                # Euclidean distance
                dist = torch.norm(coordinates[i] - coordinates[j]).item()
                edge_attr.append(dist)
    
    edge_index = torch.tensor(edge_index).t().contiguous()
    edge_attr = torch.tensor(edge_attr)
    
    # Create distance matrix for tour cost computation
    dist_matrix = torch.cdist(coordinates.unsqueeze(0), coordinates.unsqueeze(0)).squeeze(0)
    
    return Data(x=coordinates, edge_index=edge_index, edge_attr=edge_attr, 
                pos=coordinates, dist_matrix=dist_matrix)

def christofides_algorithm(data):
    """
    Run Christofides algorithm on a single TSP instance
    """
    # Create NetworkX graph from the data
    G = to_networkx(data, edge_attrs=['edge_attr'], to_undirected=True)
    
    # Rename edge attribute to 'weight' for NetworkX
    for u, v, d in G.edges(data=True):
        d['weight'] = d['edge_attr']
        del d['edge_attr']
    
    # Run Christofides algorithm
    tour = nx.approximation.christofides(G)[:-1]  # Remove duplicate last node
    tour = np.array(tour)
    
    return [tour, np.roll(tour, -1)]

def christofides_baseline(data_list):
    """
    Run Christofides baseline on a list of TSP instances
    Returns only absolute tour costs
    
    Args:
        data_list: List of Data objects or Batch object
    
    Returns:
        Tuple of (mean_cost, std_cost) for absolute tour costs
    """
    if isinstance(data_list, Batch):
        # Convert batch to list of individual data objects
        data_list = data_list.to_data_list()
    
    tour_lengths = []
    
    for i, data in enumerate(data_list):
        print(f"Processing instance {i+1}/{len(data_list)}")
        
        try:
            # Run Christofides algorithm
            tour_result = christofides_algorithm(data)
            tour = torch.tensor(tour_result[0]).long()
            
            # Calculate tour cost using distance matrix
            tour_cost = 0
            num_cities = len(tour)
            for j in range(num_cities):
                current_city = tour[j]
                next_city = tour[(j + 1) % num_cities]
                tour_cost += data.dist_matrix[current_city, next_city].item()
            
            tour_lengths.append(tour_cost)
                
        except Exception as e:
            print(f"Error processing instance {i}: {e}")
            continue
    
    return (mean(tour_lengths), std(tour_lengths))

def load_and_process_tsp_dataset(filepath):
    """
    Load TSP dataset and convert to format suitable for Christofides algorithm
    """
    print(f"Loading dataset from: {filepath}")
    
    # Load the dataset
    data = torch.load(filepath)
    coordinates = data['coordinates']
    num_instances, num_cities = coordinates.shape[:2]
    
    print(f"Loaded {num_instances} instances with {num_cities} cities each")
    
    # Convert to list of Data objects
    data_list = []
    
    for i in range(num_instances):
        coords = coordinates[i]  # (N, 2)
        
        # Convert to complete graph data
        graph_data = coords_to_complete_graph_data(coords)
        data_list.append(graph_data)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{num_instances} instances")
    
    print(f"Dataset processing complete!")
    return data_list

def main():
    """
    Main function to run Christofides baseline on TSP dataset
    """
    # Load and process the dataset
    dataset_path = '../data/tsp_200_uniform_test.pt'
    
    if not os.path.exists(dataset_path):
        print(f"Dataset file not found: {dataset_path}")
        print("Please make sure the file exists in the specified path")
        return
    
    try:
        # Load and process dataset
        data_list = load_and_process_tsp_dataset(dataset_path)
        
        # Run Christofides baseline
        print("\n" + "="*50)
        print("Running Christofides Baseline")
        print("="*50)
        
        # Run on subset for testing (first 10 instances)
        test_data = data_list[:10]
        print(f"Testing on first {len(test_data)} instances...")
        
        mean_cost, std_cost = christofides_baseline(test_data)
        
        print(f"\nTest Results (first {len(test_data)} instances):")
        print(f"Average Tour Cost: {mean_cost:.2f}")
        print(f"Standard Deviation: {std_cost:.2f}")
        
        # Run on full dataset if desired
        print(f"\nRunning on full dataset ({len(data_list)} instances)...")
        mean_cost, std_cost = christofides_baseline(data_list)
        
        print(f"\nFull Dataset Results:")
        print(f"Average Tour Cost: {mean_cost:.2f}")
        print(f"Standard Deviation: {std_cost:.2f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
