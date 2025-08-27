"""
Simple TSP data generator with 2 features (x, y coordinates only)
Generate, save, and load data for SCT model
"""

import torch
import numpy as np
import os

def generate_tsp_data(num_instances, num_cities, coord_range=100.0, seed=None):
    """
    Generate random TSP instances with just coordinates
    
    Args:
        num_instances: Number of TSP instances
        num_cities: Number of cities per instance
        coord_range: Coordinate range [0, coord_range]
        seed: Random seed for reproducibility
    
    Returns:
        coordinates: (num_instances, num_cities, 2) tensor
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Generate random coordinates
    coordinates = torch.rand(num_instances, num_cities, 2) * coord_range
    
    return coordinates

def coords_to_distance_matrix(coordinates):
    """Convert coordinates to distance matrix"""
    # coordinates: (B, N, 2) -> distance_matrix: (B, N, N)
    distance_matrix = torch.cdist(coordinates, coordinates)
    return distance_matrix
def save_tsp_dataset(coordinates, filepath, metadata=None):
    """
    Save TSP dataset to file
    
    Args:
        coordinates: (B, N, 2) coordinate tensor
        filepath: Path to save file
        metadata: Optional metadata dictionary
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    # Prepare data to save
    save_data = {
        'coordinates': coordinates,
        'num_instances': coordinates.shape[0],
        'num_cities': coordinates.shape[1],
        'coord_range': coordinates.max().item()
    }
    
    if metadata:
        save_data.update(metadata)
    
    # Save to file
    torch.save(save_data, filepath)
    
    print(f"Dataset saved to: {filepath}")
    print(f"Shape: {coordinates.shape}")
    print(f"Instances: {coordinates.shape[0]}, Cities: {coordinates.shape[1]}")

def load_tsp_dataset(filepath):
    """
    Load TSP dataset from file
    
    Args:
        filepath: Path to dataset file
    
    Returns:
        coordinates: (B, N, 2) coordinate tensor
        metadata: Dictionary with dataset info
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    
    data = torch.load(filepath, weights_only=False)
    coordinates = data['coordinates']
    
    print(f"Dataset loaded from: {filepath}")
    print(f"Shape: {coordinates.shape}")
    print(f"Instances: {data['num_instances']}, Cities: {data['num_cities']}")
    
    return coordinates, data

class SimpleTSPDataset:
    """
    Simple dataset class that only returns coordinate tensors
    """
    
    def __init__(self, coordinates):
        """
        Args:
            coordinates: (B, N, 2) coordinate tensor
        """
        self.coordinates = coordinates
        
        print(f"Creating simple TSP dataset...")
        print(f"Coordinates: {coordinates.shape}")
    
    def __len__(self):
        return len(self.coordinates)
    
    def __getitem__(self, idx):
        """Return single coordinate tensor"""
        return self.coordinates[idx]  # (N, 2)
    
    def get_batch(self, indices):
        """Get a batch of coordinate tensors"""
        return self.coordinates[indices]  # (batch_size, N, 2)

class SimpleTSPDataLoader:
    """Simple dataloader that returns coordinate tensors"""
    
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(dataset)
        self.num_batches = (self.num_samples + batch_size - 1) // batch_size
    
    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.num_samples)
        else:
            indices = torch.arange(self.num_samples)
        
        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield self.dataset.get_batch(batch_indices)  # Returns (batch_size, N, 2)
    
    def __len__(self):
        return self.num_batches

def create_standard_datasets():
    """Create standard train/val/test datasets"""
    
    print("=== Creating Standard TSP Datasets (2 features only) ===")
    
    # Parameters
    num_cities = 20
    coord_range = 100.0
    
    # Create train set
    print("\n1. Generating training set...")
    train_coords = generate_tsp_data(
        num_instances=200000,  # Reduced for challenging distributions, 200000 for 20, 50,0000 for 50, 1500000 for 100
        num_cities=num_cities,
        coord_range=coord_range,
        seed=42
    )
    save_tsp_dataset(train_coords, f'data/tsp_{num_cities}_uniform_train.pt', 
                     {'split': 'train', 'description': 'Training set with 2 features (x,y)'})
    
    # Create validation set
    print("\n2. Generating validation set...")
    val_coords = generate_tsp_data(
        num_instances=1000,
        num_cities=num_cities,
        coord_range=coord_range,
        seed=123
    )
    save_tsp_dataset(val_coords, f'data/tsp_{num_cities}_uniform_val.pt',
                     {'split': 'val', 'description': 'Validation set with 2 features (x,y)'})
    
    # Create test set
    print("\n3. Generating test set...")
    test_coords = generate_tsp_data(
        num_instances=1000,
        num_cities=num_cities,
        coord_range=coord_range,
        seed=456
    )
    save_tsp_dataset(test_coords, f'data/tsp_{num_cities}_uniform_test.pt',
                     {'split': 'test', 'description': 'Test set with 2 features (x,y)'})
    
    print("\n=== Dataset Creation Complete ===")
    return train_coords, val_coords, test_coords

def test_dataset_loading():
    """Test loading and using the datasets"""
    print("\n=== Testing Dataset Loading ===")
    
    # Load train dataset
    train_coords, train_metadata = load_tsp_dataset('data/tsp_train.pt')
    
    # Create simple dataset
    dataset = SimpleTSPDataset(train_coords[:100])  # Use first 100 for testing
    
    # Create dataloader
    dataloader = SimpleTSPDataLoader(dataset, batch_size=16, shuffle=True)
    
    print(f"\nDataloader test:")
    print(f"Number of batches: {len(dataloader)}")
    
    # Test first batch
    for batch_idx, coordinates_batch in enumerate(dataloader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Coordinates batch: {coordinates_batch.shape}")  # Should be (batch_size, N, 2)
        
        # Verify shape
        assert coordinates_batch.dim() == 3, "Should be 3D tensor (batch_size, N, 2)"
        assert coordinates_batch.shape[2] == 2, "Should have 2 features (x, y)"
        print(f"  ✓ Correct shape: {coordinates_batch.shape}")
        
        if batch_idx == 0:  # Only test first batch
            break
    
    print("\n✓ Dataset loading test passed!")

if __name__ == "__main__":
    # Create the datasets
    train_coords, val_coords, test_coords = create_standard_datasets()
    
    # Test loading
    # test_dataset_loading()
    
    # Show final stats
    print(f"\n=== Final Dataset Statistics ===")
    print(f"Training set: {train_coords.shape}")
    print(f"Validation set: {val_coords.shape}")
    print(f"Test set: {test_coords.shape}")
    print(f"Features per node: 2 (x, y coordinates only)")
    print(f"Coordinate range: [0, {train_coords.max():.1f}]")
