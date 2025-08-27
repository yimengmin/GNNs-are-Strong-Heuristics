import torch
'''
from https://github.com/yimengmin/UTSP
'''
def tsp_permutation_loss(nn_output, distance_matrix, shift=-1):
    '''
    input:
    nn_output: batchsize * num_of_nodes * num_of_nodes tensor
    distance_matrix: batchsize * num_of_nodes * num_of_nodes tensor
    shift: integer shift parameter for rolling the transpose matrix
    '''
    # Create heat map using matrix multiplication with rolled transpose
    heat_map = torch.matmul(nn_output, torch.roll(torch.transpose(nn_output, 1, 2), shift, 1))

    # Calculate weighted path length
    weighted_path = torch.mul(heat_map, distance_matrix)
    weighted_path = weighted_path.sum(dim=(1,2))

    return weighted_path, heat_map

