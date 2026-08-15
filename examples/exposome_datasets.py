# copy the license file etc
'''
Dataset wrapper for exposome PyG graphs
'''

import torch
import time
from torch.utils.data import Dataset

class ExposomeDataset(Dataset):

  def __init__(self, data):

    self.data = data

  def __len__(self):

    return 1

  def __getitem__(self, index):

    if index != 0:
      raise IndexError("This dataset contains one graph.")

    self.data.y = self.data.x

    return self.data

# open the pt file
graph_file = '/lustre/orion/med117/proj-shared/exposomedata/test_static/static/static_normalized_k1_h3_08.pt'

start = time.perf_counter()

graphs = torch.load( graph_file, weights_only=False, map_location="cpu" )

elapsed= time.perf_counter() - start

print( f'Done loading the graph: {elapsed}' )

print( graphs )

dataset = ExposomeDataset( graphs )

print(dataset.__getitem__(0))
