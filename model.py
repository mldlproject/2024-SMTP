import torch
import math
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.functional import leaky_relu
from torch_geometric.nn import Set2Set
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.nn import GCNConv
from torch.nn.init import kaiming_uniform_, zeros_
from torch_geometric.nn import global_max_pool

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

class MultiHeadTripletAttention(MessagePassing):
    def __init__(self, node_channels, edge_channels, heads=3, negative_slope=0.2, **kwargs):
        super(MultiHeadTripletAttention, self).__init__(aggr='add', node_dim=0, **kwargs)  # aggr='mean'
        self.node_channels = node_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.weight_node = Parameter(torch.Tensor(node_channels, heads * node_channels))
        self.weight_edge = Parameter(torch.Tensor(edge_channels, heads * node_channels))
        self.weight_triplet_att = Parameter(torch.Tensor(1, heads, 3 * node_channels))
        self.weight_scale = Parameter(torch.Tensor(heads * node_channels, node_channels))
        self.bias = Parameter(torch.Tensor(node_channels))
        self.reset_parameters()

    def reset_parameters(self):
        kaiming_uniform_(self.weight_node)
        kaiming_uniform_(self.weight_edge)
        kaiming_uniform_(self.weight_triplet_att)
        kaiming_uniform_(self.weight_scale)
        zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr, size=None):
        x = torch.matmul(x, self.weight_node)
        edge_attr = torch.matmul(edge_attr, self.weight_edge)
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

    def message(self, x_j, x_i, edge_index_i, edge_attr, size_i):
        # Compute attention coefficients.
        x_j = x_j.view(-1, self.heads, self.node_channels)
        x_i = x_i.view(-1, self.heads, self.node_channels)
        e_ij = edge_attr.view(-1, self.heads, self.node_channels)

        triplet = torch.cat([x_i, e_ij, x_j], dim=-1)  # time consuming 13s
        alpha = (triplet * self.weight_triplet_att).sum(dim=-1)  # time consuming 12.14s
        alpha = leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, ptr=None, num_nodes=size_i)
        alpha = alpha.view(-1, self.heads, 1)

        return alpha * e_ij * x_j

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.heads * self.node_channels)
        aggr_out = torch.matmul(aggr_out, self.weight_scale)
        aggr_out = aggr_out + self.bias
        return aggr_out

    def extra_repr(self):
        return '{node_channels}, {node_channels}, heads={heads}'.format(**self.__dict__)



class Block(torch.nn.Module):
    def __init__(self, dim, edge_dim, heads=4, time_step=3):
        super(Block, self).__init__()
        self.time_step = time_step
        self.conv = MultiHeadTripletAttention(dim, edge_dim, heads)  # GraphMultiHeadAttention
        self.ln = nn.LayerNorm(dim)

    def forward(self, x, edge_index, edge_attr):
        m = F.celu(self.conv.forward(x, edge_index, edge_attr))
        x = self.ln(m.squeeze(0))
        return x


class STP_Model(torch.nn.Module):
    def __init__(self, in_dim, edge_in_dim, hidden_dim=32, depth=3, heads=4, dropout=0.1, outdim=1):
        super(STP_Model, self).__init__()
        self.depth = depth
        self.dropout = dropout
        # 
        self.conv1  = GCNConv(in_dim, hidden_dim*2)
        self.conv2  = GCNConv(hidden_dim*2, hidden_dim*4)
        self.conv3  = GCNConv(hidden_dim*4, 300)
        
        self.message_pharse  = Block(300, edge_in_dim, heads)

        self.global_max_pool = global_max_pool
        
        self.out = nn.Sequential(
            nn.Linear(300, 35),
        )

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        x = F.relu(self.conv3(x, data.edge_index))
    
        x = x + F.dropout(self.message_pharse(x, data.edge_index, data.edge_attr))
        features_of_masked_atoms = x[data.masked_atom_indices] # select masked atoms
        x = self.out(F.dropout(features_of_masked_atoms))
        return x

    def extract_features(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        x = F.relu(self.conv3(x, data.edge_index))
        x = x + F.dropout(self.message_pharse(x, data.edge_index, data.edge_attr))
        x = self.global_max_pool(x, data.batch)
        return x