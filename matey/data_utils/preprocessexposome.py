import os, argparse
import polars as pl
import h3.api.numpy_int as h3_np
import math
import torch
from torch_geometric.data import Data
import numpy as np

EARTH_RADIUS_M = 6371008.8
def _edge_haversine_m(pos, edge_index, nedge_half, chunk_size=2_000_000):
    """
    pos:        np.ndarray [N, 2], float32, (lat, lon) in degrees
    edge_index: torch.LongTensor [2, 2*E_half]
    """
    edge_dist = torch.empty((2 * nedge_half, 1), dtype=torch.float32)

    src_all = edge_index[0, :nedge_half].numpy()
    dst_all = edge_index[1, :nedge_half].numpy()

    for start in range(0, nedge_half, chunk_size):
        end = min(start + chunk_size, nedge_half)

        src = src_all[start:end]
        dst = dst_all[start:end]

        lat1 = np.deg2rad(pos[src, 0])
        lon1 = np.deg2rad(pos[src, 1])
        lat2 = np.deg2rad(pos[dst, 0])
        lon2 = np.deg2rad(pos[dst, 1])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

        # Avoid tiny floating point excursions above 1.
        np.clip(a, 0.0, 1.0, out=a)

        dist = (2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))).astype(np.float32, copy=False)

        dist_t = torch.from_numpy(dist)

        # forward
        edge_dist[start:end, 0] = dist_t

        # reverse -- same distance
        edge_dist[nedge_half + start:nedge_half + end, 0] = dist_t

    return edge_dist

def _haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
    return 2 * R * math.asin(math.sqrt(a))
def _cell_latlng(h3, cell):
    if hasattr(h3, "cell_to_latlng"):
        return h3.cell_to_latlng(cell) # (lat, lng)
    if hasattr(h3, "h3_to_geo"):
        return h3.h3_to_geo(cell) # (lat, lng)

def h3_cells_to_pyg_instant(dataobj, k_cutoff = 1, level='h3_08'):
    """
    #NOTE: the current parquet data version by default is level 8, so level is not used for now
    """
    node_chunk_size=250_000
    distance_chunk_size=2_000_000
    #get metadata from the first row
    feature_cols = [c for c in dataobj.columns if c != "h3_index"]
    nrows = len(dataobj)
    print(f"[H3 graph] nodes={nrows:,}, features={len(feature_cols)}, k={k_cutoff}", flush=True)
    print(f"[H3 graph] feature columns: {feature_cols}", flush=True)

    cells = np.fromiter((int(c, 16) for c in dataobj["h3_index"]), dtype=np.uint64, count=nrows)
    num_nodes = cells.size
    if num_nodes == 0:
        raise ValueError("No H3 cells found")

    sort_idx = np.argsort(cells)
    sorted_cells = cells[sort_idx]
    duplicate_mask = sorted_cells[1:] == sorted_cells[:-1]
    if np.any(duplicate_mask):
        ndup = int(np.count_nonzero(duplicate_mask))
        raise ValueError(f"Unexpected duplicate H3 cells: found at least {ndup:,} duplicate entries")

    # Maximum number of neighbors excluding self in a regular hexagonal grid disk.
    max_neighbors = 3 * k_cutoff * (k_cutoff + 1)
    half_edge_chunks = []
    total_half_edges = 0

    for block_start in range(0, num_nodes, node_chunk_size):
        block_end = min(block_start + node_chunk_size, num_nodes)
        nblock = block_end - block_start
        # Maximum candidates for this block.
        max_candidates = nblock * max_neighbors

        src_candidate = np.empty(max_candidates, dtype=np.int64)
        nbr_candidate = np.empty(max_candidates, dtype=np.uint64)

        ncandidate = 0
        for i in range(block_start, block_end):
            cell = cells[i]
            nbrs = h3_np.grid_disk(cell, k_cutoff) # distance <= k_cutoff (includes cell)
            # Remove self.
            nbrs = nbrs[nbrs != cell]
            n = nbrs.size
            if n == 0:
                continue

            src_candidate[ncandidate:ncandidate + n] = i
            nbr_candidate[ncandidate:ncandidate + n] = nbrs
            ncandidate += n

        src_candidate = src_candidate[:ncandidate]
        nbr_candidate = nbr_candidate[:ncandidate]

        p = np.searchsorted(sorted_cells, nbr_candidate)
        valid = p < num_nodes
        p = p[valid]
        src = src_candidate[valid]
        nbr = nbr_candidate[valid]

        # Neighbor must actually exist in this dataset.
        hit = sorted_cells[p] == nbr
        p = p[hit]
        src = src[hit]

        # Convert position in sorted array -> original graph index.
        dst = sort_idx[p]

        
        # Keep only one copy of each undirected edge.
        keep = src < dst

        src = src[keep]
        dst = dst[keep]

        n_half = src.size

        if n_half > 0:
            edge_chunk = np.empty((2, n_half), dtype=np.int64)

            edge_chunk[0] = src
            edge_chunk[1] = dst

            half_edge_chunks.append(torch.from_numpy(edge_chunk))

            total_half_edges += n_half

        print(f"[H3 graph] processed nodes {block_end:,}/{num_nodes:,};  unique edges={total_half_edges:,}", flush=True)

    del sorted_cells
    del sort_idx

    half_edge_index = torch.cat(half_edge_chunks, dim=1)
    del half_edge_chunks
    nedge_half = half_edge_index.shape[1]
    edge_index = torch.empty((2, 2 * nedge_half), dtype=torch.long)

    # forward
    edge_index[:, :nedge_half] = half_edge_index
    # reverse
    edge_index[0, nedge_half:] = half_edge_index[1]
    edge_index[1, nedge_half:] = half_edge_index[0]

    del half_edge_index
    print(f"[H3 graph] unique undirected edges={nedge_half:,}; PyG directed edges={edge_index.shape[1]:,}", flush=True)

    #node positions
    pos_np = np.empty((num_nodes, 2), dtype=np.float32)
    for block_start in range(0, num_nodes, node_chunk_size):
        block_end = min(block_start + node_chunk_size, num_nodes)
        for i in range(block_start, block_end):
            lat, lon = h3_np.cell_to_latlng(cells[i])
            pos_np[i, 0] = lat
            pos_np[i, 1] = lon
        print(f"[H3 graph] positions  {block_end:,}/{num_nodes:,}", flush=True)

    
    #Edge distances
    edge_dist = _edge_haversine_m(pos_np, edge_index, nedge_half, chunk_size=distance_chunk_size)

    # Node features
    # x itself is ~1.5 GB for 10.7M x 37 float32 values.

    x_np = dataobj.select([pl.col(c).cast(pl.Float32) for c in feature_cols]).to_numpy()

    x_np = np.ascontiguousarray(x_np, dtype=np.float32)
    x = torch.from_numpy(x_np)

    # Construct Data
    
    graph_kwargs = {
        "x": x,
        "edge_index": edge_index,
        "num_nodes": num_nodes,
        "h3_cells": torch.from_numpy(cells.view(np.int64)),
        "nodefeatures": feature_cols
        }

    graph_kwargs["pos"] = torch.from_numpy(pos_np)

    if edge_dist is not None:
        graph_kwargs["edge_attr"] = edge_dist

    graph = Data(**graph_kwargs)

    print(graph, flush=True)
    

    return graph
    
def h3_cells_to_pyg(dataobj, outputname, k_cutoff = 1, level='h3_08'):
    feature_cols = [c for c in dataobj.columns if c != "h3_index"]
    nrows = len(dataobj)
    print(f"[H3 graph] total spacetime nodes={nrows:,}, features={len(feature_cols)}, k={k_cutoff}", flush=True)
    print(f"[H3 graph] feature columns: {feature_cols}", flush=True)
    time_col = dataobj["time"].unique().sort().to_numpy()
    assert len(time_col)==7, f"[H3 graph] expecting 7 time points (for a week) every file but got {len(time_col)}: {time_col}"
    #assert nrows%len(time_col)==0, f"[H3 graph] expecting {len(time_col)} time points for each unique h3 cell but got {nrows} rows"
    for time in time_col:
        rawdata = dataobj.filter(pl.col("time")==time)
        graph = h3_cells_to_pyg_instant(rawdata, k_cutoff=k_cutoff, level=level)
        torch.save(graph, outputname[:-3]+f"_{time}.pt")

def graph_convert(rawdata, outputname, k_cutoff = 1, level='h3_08',timescale='static'):
    if timescale == 'static':   
        #10664999 X 38
        #['h3_index', 'elevation', 'land_cover', 'day_population', 'night_population', 'a_k', 'c_k', 'top5_k', 
        # 'a_th', 'a_u', 'a_10a_cly', 'a_14a_cly', 'a_klnt', 'c_th', 'c_u', 'top5_th', 'top5_u', 'c_10a_cly', 
        # 'c_14a_cly', 'c_klnt', 'count', 'aqpermnew', 'slope', 'tave', 'ppt', 'pet', 'sand', 'pmpe', 'minele', 
        # 'relief', 'pflattot', 'pflatlow', 'pflatup', 'hlr', 'logk_ferr_', 'logk_ice_x', 'k_stdev_x1', 'porosity_x']
        graph = h3_cells_to_pyg_instant(rawdata, k_cutoff=k_cutoff, level=level)
        torch.save(graph, outputname)
    else:
        #70M X 15
        #['h3_index', 'shortwave_radiation', 'day_length', 'snow_water_equivalent', 'vapour_pressure', 'precipitation', 
        # 'max_air_temperature', 'min_air_temperature', 'date', 'wind_speed', 'wind_from_direction', 'min_relative_humidity', 
        # 'max_relative_humidity', 'time', 'smokepm_pred']
        #NOTE: need to check if we are expecting 7 times 10,664,999 grid points for each file, or if the grid points vary by time
        graph = h3_cells_to_pyg(rawdata, outputname, k_cutoff=k_cutoff, level=level)
    
    return
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default='/lustre/orion/med117/proj-shared/exposomedata/exposome_fm_test_08042026', help='Directory containing the data')
    parser.add_argument('--output_dir', default='/lustre/orion/med117/proj-shared/exposomedata/exposome_fm_test_08042026_prebuiltgraphs')
    parser.add_argument('--timescale', default='static', choices=['static', 'daily', 'yearly'], help='Timescale for the data')
    parser.add_argument('--k_cutoff', type=int, default=1)
    parser.add_argument('--level', type=str, default='h3_08')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    args = parser.parse_args()

    rawdata_dir = os.path.join(args.datadir, args.timescale)
    output_dir = os.path.join(args.output_dir, args.timescale) 

    os.makedirs(output_dir, exist_ok=True)
    rawfiles = [f for f in os.listdir(rawdata_dir) if f.endswith('.parquet')]  
    nfiles = len(rawfiles)

    for ifile, filename in enumerate(rawfiles):
        filepath = os.path.join(rawdata_dir, filename)
        #get fiel name without path and extension
        filename = os.path.splitext(filename)[0]
        output_file = os.path.join(output_dir, f"{filename}_k{args.k_cutoff}_{args.level}.pt")
        print(filepath, output_file)
        if os.path.exists(output_file) and not args.overwrite:
            continue
        print(f"Processing {filepath} --> {output_file}", flush=True)
        rawdata = pl.read_parquet(filepath)#.head(500000)
        print(rawdata.head())
        graph_convert(rawdata, output_file, k_cutoff=args.k_cutoff, level=args.level, timescale=args.timescale)


    

    
