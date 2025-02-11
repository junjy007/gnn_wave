import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from config import Config

class GNN(torch.nn.Module):
    def __init__(self, cfg:Config):
        super(GNN, self).__init__()
        self.config = cfg # keep a reference of the configuration
        self.num_input_variables = len(cfg.node_var_observ)* cfg.snapshots
        self.num_output_variables = len(cfg.node_var_target)
        self.conv_layers = torch.nn.ModuleList()

        # Create convolutional layers based on cfg.convolution_kernels
        conv_sizes = [self.num_input_variables] + list(cfg.convolution_kernels)
        for i in range(len(conv_sizes) - 1):
            self.conv_layers.append(GCNConv(conv_sizes[i], conv_sizes[i+1]))
        self.out = torch.nn.Linear(cfg.convolution_kernels[-1], self.num_output_variables)    
        
    def forward(self, x, edge_index, x_bc=None, bc_location_indices=None):
        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = F.relu(x)
       
        pred = self.out(x)
        # we can reduce the output to be located to the self.config.prediction_range
        # trim pred using the range.
        return pred

