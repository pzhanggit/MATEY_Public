# copy the license file etc
'''
Dataset wrapper for exposome PyG graphs
'''

import torch
import time
from torch.utils.data import Dataset
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data


class ExposomeDataset(Dataset):

  # the function parameter is the global graph
  def __init__(self, data, k ):

    self.global_graph = data

    self.k = k

  def __len__(self):

    return len( self.global_graph )

  # this returns the k-hop subgraph reachable from starting node at "index" in the global graph
  def __getitem__(self, index):

    if index >= len( self.global_graph.x ):

      raise IndexError("Out of range index requested! Exiting...")

    # discover the indices needed from global_graph to build the k-hop subgraph
    subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=index,
        num_hops=self.k,
        edge_index=self.global_graph.edge_index,
        relabel_nodes=True,
        num_nodes=self.global_graph.num_nodes,
        directed=False,
        )

    # use the indices above to extract the subgraph from the global_graph
    subgraph = Data(
          x=self.global_graph.x[subset],
          edge_index=sub_edge_index,
          edge_attr=self.global_graph.edge_attr[edge_mask] if self.global_graph.edge_attr is not None else None,
          y=self.global_graph.y[subset] if getattr(self.global_graph, "y", None) is not None else None,
          pos=self.global_graph.pos[subset] if getattr(self.global_graph, "pos", None) is not None else None,
          num_nodes=subset.numel(),
          )

    # original node indices from global graph; the are remapped in the subgraph
    subgraph.n_id = subset

    # the nodes in the subgraph have been remapped to local indices; this is the local
    # index of the center node that was requested from the global graph
    center_local_idx = mapping

    return subgraph

# main function:

# open the pt file
graph_file = '/lustre/orion/med117/proj-shared/exposomedata/test_static/static/static_normalized_k1_h3_08.pt'

start = time.perf_counter()

# load the global graph
graphs = torch.load( graph_file, weights_only=False, map_location="cpu" )

elapsed= time.perf_counter() - start

print( f'Done loading the graph: {elapsed} seconds' )

# show dimensionality of the global graph
print( graphs )

# number of hops to extract around the node at index in the global_graph
# this should be equal to the number of layers in the GNN
k = 1

# create the dataset
dataset = ExposomeDataset( graphs, k )

# arbitrarily pick a node somewhere in the middle of the global list
index = 5332500

start = time.perf_counter()
# extract the subgraph and print it
print(dataset.__getitem__( index ))

elapsed= time.perf_counter() - start

print( f'Done extracting the subgraph: {elapsed} seconds' )



























