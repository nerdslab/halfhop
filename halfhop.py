import copy

import torch
from torch_geometric.utils import subgraph



class HalfHop:
    r"""Graph upsampling augmentation. Adds artifical slow nodes between neighbors to 
    slow down message propagation.

    ..note:: Use the :obj:`slow_node_mask` attribute to mask out the slow nodes after
    message passing.
    
    Args:
        alpha (float): The interpolation factor for the slow nodes.
        p (float): The probability of half-hopping an edge.
        inplace (bool): If set to :obj:`False`, will not modify the input graph
            and will instead return a new graph.
    """
    def __init__(self, alpha=0.5, p=1.0, inplace=True):
        assert 0. <= p <= 1., f"p must be in [0, 1], got {p}"
        assert 0. <= alpha <= 1., f"alpha must be in [0, 1], got {alpha}"

        self.p = p
        self.alpha = alpha

        self.inplace = inplace

    def __call__(self, data):
        if not self.inplace:
            data = copy.deepcopy(data)

        x, edge_index = data.x, data.edge_index
        device = data.x.device

        # first, isolate self loops which are not half-hopped
        self_loop_mask = edge_index[0] == edge_index[1]
        edge_index_self_loop = edge_index[:, self_loop_mask]
        edge_index = edge_index[:, ~self_loop_mask]

        # decide which edges to half-hop
        if self.p == 1.:
            # all edges are half-hopped
            edge_index_to_halfhop = edge_index
            edge_index_to_keep = None
        else:
            # randomly sample nodes and half-hop their edges
            node_mask = torch.rand(data.num_nodes, device=device) < self.p
            _, _, edge_mask = subgraph(node_mask, torch.stack([edge_index[1], edge_index[1]], dim=0), return_edge_mask=True)
            edge_index_to_halfhop = edge_index[:, edge_mask]
            edge_index_to_keep = edge_index[:, ~edge_mask]

        # add new slow nodes, and use linear interpolation to initialize their features
        slow_node_ids = torch.arange(edge_index_to_halfhop.size(1), device=device) + data.num_nodes
        x_slow_node = x[edge_index_to_halfhop[0]]
        x_slow_node.mul_(self.alpha).add_(x[edge_index_to_halfhop[1]], alpha=1. - self.alpha)
        new_x = torch.cat([x, x_slow_node], dim=0)

        # add new edges between slow nodes and the original nodes that replace the original edges
        edge_index_slow = [
            torch.stack([edge_index_to_halfhop[0], slow_node_ids]),
            torch.stack([slow_node_ids, edge_index_to_halfhop[1]]),
            torch.stack([edge_index_to_halfhop[1], slow_node_ids])
            ]
        new_edge_index = torch.cat([edge_index_to_keep, edge_index_self_loop, *edge_index_slow], dim=1)

        # prepare a mask that distinguishes between original nodes and slow nodes
        slow_node_mask = torch.cat([
            torch.zeros(x.size(0), device=device),
            torch.ones(slow_node_ids.size(0), device=device)
        ], dim=0).bool()

        data.x, data.edge_index, data.slow_node_mask = new_x, new_edge_index, slow_node_mask

        return data

    def __repr__(self):
        return '{}(alpha={}, p={})'.format(self.__class__.__name__, self.alpha, self.p)
