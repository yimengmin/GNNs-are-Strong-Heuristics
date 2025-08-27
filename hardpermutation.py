import torch
from scipy.optimize import linear_sum_assignment
import numpy as np

def to_exact_permutation(soft_perm_matrix, temperature=1.0):
    """
    Convert a soft permutation matrix to an exact permutation matrix.
    
    Args:
        soft_perm_matrix: Output from your model (batch_size, n, n)
        temperature: Temperature for sharpening the distribution
    
    Returns:
        Exact permutation matrix (batch_size, n, n)
    """
    soft_perm_matrix = soft_perm_matrix.to(torch.float32)
    batch_size, n, _ = soft_perm_matrix.shape
    device = soft_perm_matrix.device
    
    # Move soft permutation matrix to CPU in one step and process as a whole batch
    soft_perm_matrix_cpu = soft_perm_matrix.detach().cpu().numpy()  # Shape: (batch_size, n, n)
    
    # Initialize output tensor for hard permutations
    hard_perms = np.zeros_like(soft_perm_matrix_cpu, dtype=np.float32)
    
    # Apply Hungarian algorithm batch-wise
    for b in range(batch_size):
        # Apply Hungarian algorithm for each matrix in the batch
        row_ind, col_ind = linear_sum_assignment(-soft_perm_matrix_cpu[b])  # Negative for maximization
        hard_perms[b][row_ind, col_ind] = 1  # Construct permutation matrix
    
    # Convert back to a torch tensor in one step
    hard_perms = torch.tensor(hard_perms, device=device)
    
    if soft_perm_matrix.requires_grad:
        # Straight-through estimator for backprop
        hard_perms = (hard_perms - soft_perm_matrix).detach() + soft_perm_matrix
    
    return hard_perms



from torch_linear_assignment import batch_linear_assignment, assignment_to_indices

def to_exact_permutation_batched(soft_perm_matrix, temperature=1.0):
    """
    Convert a soft permutation matrix to an exact permutation matrix using torch_linear_assignment.

    Args:
        soft_perm_matrix: Output from your model (batch_size, n, n)
        temperature: Temperature for sharpening the distribution (currently unused)

    Returns:
        Exact permutation matrix (batch_size, n, n)
    """
    soft_perm_matrix = soft_perm_matrix.to(torch.float32)
    batch_size, n, _ = soft_perm_matrix.shape
    device = soft_perm_matrix.device

    # Negative for maximization (converting similarity to cost)
    cost_matrix = -soft_perm_matrix

    # Get assignments using batch_linear_assignment
    assignments = batch_linear_assignment(cost_matrix)

    # Convert assignments to row and column indices
    row_ind, col_ind = assignment_to_indices(assignments)

    # Create empty tensor for hard permutation matrices
    hard_perms = torch.zeros_like(soft_perm_matrix)

    # Convert indices to permutation matrices
    # Using advanced indexing to set values to 1 at assigned positions
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, row_ind.size(1))
    hard_perms[batch_indices, row_ind, col_ind] = 1

    if soft_perm_matrix.requires_grad:
        # Straight-through estimator for backprop
        hard_perms = (hard_perms - soft_perm_matrix).detach() + soft_perm_matrix

    return hard_perms


'''
def to_exact_permutation(soft_perm_matrix, temperature=1.0):
    """
    Convert a soft permutation matrix to an exact permutation matrix.
    
    Args:
        soft_perm_matrix: Output from your model (batch_size, n, n)
        temperature: Temperature for sharpening the distribution
    
    Returns:
        Exact permutation matrix (batch_size, n, n)
    """
    batch_size, n, _ = soft_perm_matrix.shape
    device = soft_perm_matrix.device
    
    # Initialize output tensor
    hard_perms = torch.zeros_like(soft_perm_matrix)
    soft_perm_matrix_cpu = soft_perm_matrix.detach().cpu().numpy() 
    # Process each matrix in the batch
    for b in range(batch_size):
        # Get the current soft permutation matrix
        #soft_perm = soft_perm_matrix[b].detach().cpu().numpy()
        soft_perm = soft_perm_matrix_cpu[b]
        
        # Apply Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(-soft_perm)  # Negative for maximization
        
        # Create exact permutation matrix
        perm = np.zeros_like(soft_perm)
        perm[row_ind, col_ind] = 1
        
        # Convert back to tensor
        hard_perms[b] = torch.tensor(perm, device=device)
    
    if soft_perm_matrix.requires_grad:
        # Straight-through estimator for backprop
        hard_perms = (hard_perms - soft_perm_matrix).detach() + soft_perm_matrix
    
    return hard_perms
'''
