# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 UT-Battelle, LLC
# This file is part of the MATEY Project.

"""Dataset wrapper for H3 sensor/query PyG graphs.

This loader consumes the `.pt` files produced by graph builder that
converts sensor/query parquet rows into PyTorch Geometric `Data` objects.
Each saved file is expected to contain a list of `torch_geometric.data.Data`
objects.  Each graph should contain at least:
    x:            [num_nodes, num_features]
    edge_index:   [2, num_edges]
    edge_attr:    optional edge features
    pos:          optional node positions
    node_type:    sensor/query node ids; query nodes are assumed to be > 0
    target_idx:   scalar index of the prediction target in x

The dataset returns the graph-format dictionary: {"graph", "bcs", "field_labels_out"}.
"""
import copy
import glob
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Data
try:
    import yaml
except ImportError:
    yaml = None
from matey.data_utils.utils import unwrap_leadtime_config, GhostInfo

import argparse, tempfile

class H3AirQualityGraphDataset(Dataset):
    """dataset for H3 air-quality sensor/query graphs.
    The default mode is same-time sensor-to-query regression: the target is read
    from the same graph snapshot and the target feature is masked on query nodes
    in the model input.  Set `same_time_target: false` in supportdata if you want
    autoregressive graph-to-graph prediction instead.
    """
    def __init__(self, path, include_string='', n_steps=1, dt=1, leadtime_config={}, supportdata=None, split='train', 
        train_val_test=None, extra_specific=False, tokenizer_heads=None, tkhead_name=None, SR_ratio=None,
        group_id=0, group_rank=0, group_size=1, use_dist=False, partition_method  = "metis"):
        
        super().__init__()
        self.path = path
        self.split = split
        self.train_val_test = train_val_test 
        assert self.train_val_test is None, f"{self.__class__.__name__} uses pre-split train/val/test datasets for now"
        self.extra_specific = extra_specific 
        self.include_string = include_string if len(include_string)>0 else split
        self.dt = dt
        assert self.dt==1, f"{self.__class__.__name__} currently only supports dt=1 but got {dt}"
        self.leadtime_max, self.leadtime_fixed, self.leadtime_returnfull = unwrap_leadtime_config(leadtime_config)
        #if leadtime_fixed == True, set leadtime for all samples to be constant leadtime_max
        self.nsteps_input = n_steps
        self.partition = {'train': 0, 'val': 1, 'test': 2}[split]

        self.tokenizer_heads = tokenizer_heads
        self.tkhead_name=tkhead_name
        self.group_id=group_id
        self.group_rank=group_rank
        self.group_size=group_size
        self.use_dist=bool(use_dist or (self.group_size>1))
        if self.use_dist:
            assert dist.is_available() and dist.is_initialized(), (
                "torch.distributed must be initialized when use_dist=True "
                "or group_size > 1"
            )

        self.partition_method = partition_method
        #TODO: placeholder, not used yet
        self.partition_root = self.path+f"/partitioned_{self.group_size}"

        self.feature_names, self.query_features, self.field_names_out, self.type, self.time_steps, self.num_node_types = self._specifics()
        self.title = self.type

        opts = self._load_support_options(supportdata)
        self.same_time_target = bool(opts.get("same_time_target", True))
        self.mask_query_target = bool(opts.get("mask_query_target", True))
        self.mask_query_lag_features = bool(opts.get("mask_query_lag_features", True))
        self.mask_value = float(opts.get("mask_value", -100.0))
        self.query_node_min_type = int(opts.get("query_node_min_type", 1))
        self.bcs = torch.as_tensor(opts.get("bcs", [0, 0]), dtype=torch.float32)

        self.config = self._load_graph_config(opts)
        self.graph_file = self._resolve_graph_file(opts)

        self.graphs = torch.load(self.graph_file, weights_only=False, map_location="cpu")
        print(len(self.graphs), self.graphs[0], flush=True)
        for graph in self.graphs:
            # Patch isolated nodes without changing the original graph builder.
            self._connect_zero_degree_nodes_to_nearest_(graph)
            graph.x = torch.cat([graph.x, graph.pos], dim=1)
        self.get_minmax()
        self.get_node_type()

        self.plot_dataset_histograms(save_dir="./",bins=50)

        if not isinstance(self.graphs, list) or len(self.graphs) == 0:
            raise ValueError(f"Expected a non-empty list of PyG Data objects in {self.graph_file}.")
        if not all(isinstance(g, Data) for g in self.graphs):
            raise TypeError(f"All objects in {self.graph_file} must be torch_geometric.data.Data instances.")

        self.target_name = self.field_names_out
        self.target_feature_idx = self._graph_target_idx(self.graphs[0])
        assert self.target_name[0]==self.feature_names[self.target_feature_idx], f"check target name {self.target_name, self.feature_names, self.target_feature_idx}"
        
        print(self.target_name, self.feature_names, self.target_feature_idx, flush=True)

        if self.same_time_target:
            self.valid_length = len(self.graphs)
        else:
            raise RuntimeError("not supporting this yet")
            # Same convention as existing graph datasets: input covers n_steps,
            # target is n_steps + leadtime - 1 after the starting index.
            lead = max(int(self.leadtime_max), 1)
            self.valid_length = len(self.graphs) - self.nsteps_input - lead + 1
        if self.valid_length < 1:
            raise ValueError(
                f"Dataset {self.graph_file} is too short for n_steps={self.nsteps_input} "
                f"and leadtime_max={self.leadtime_max}."
            )

    @staticmethod
    def _specifics():
        type = "h3airqualitygraph"
        nlags=15
        ##131 node features + 3 pos
        feature_names = (
            ["conc", "min_air_temperature"]
            + [f"lags_min_air_temperature_{it}" for it in range(1, 1 + nlags)]
            + ["max_air_temperature"]
            + [f"lags_max_air_temperature_{it}" for it in range(1, 1 + nlags)]
            + ["min_relative_humidity"]
            + [f"lags_min_relative_humidity_{it}" for it in range(1, 1 + nlags)]
            + ["max_relative_humidity"]
            + [f"lags_max_relative_humidity_{it}" for it in range(1, 1 + nlags)]
            + ["wind_speed"]
            + [f"lags_wind_speed_{it}" for it in range(1, 1 + nlags)]
            + ["precipitation_amount"]
            + [f"lags_precipitation_amount_{it}" for it in range(1, 1 + nlags)]
            + ["wind_direction"]
            + [f"lags_wind_direction_{it}" for it in range(1, 1 + nlags)]
            + [f"lags_pm_prior_{it}" for it in range(1, 1 + nlags)]
            + ["day_population", "night_population", "land_cover"]
            + ["lat", "lon", "elevation"]
        )
        query_features = (
            ["min_air_temperature", "max_air_temperature", "min_relative_humidity", "max_relative_humidity"]
            + ["wind_speed", "precipitation_amount", "wind_direction"]
            + ["day_population", "night_population", "land_cover"]
            + ["lat", "lon", "elevation"]
        )
        field_names_out = ["conc"]
        time_steps=601
        num_node_types = 5
        return feature_names, query_features, field_names_out, type, time_steps, num_node_types
    field_names = _specifics()[0]

    def get_name(self):
        return self.type

    def get_minmax(self):
        # Calculate min/max of each feature in graph.x across all graphs and all nodes.
        feat_min = None
        feat_max = None

        for g in self.graphs:
            x = g.x.float()  # shape: [num_nodes, num_features]

            cur_min = x.amin(dim=0)
            cur_max = x.amax(dim=0)

            feat_min = cur_min if feat_min is None else torch.minimum(feat_min, cur_min)
            feat_max = cur_max if feat_max is None else torch.maximum(feat_max, cur_max)

        self.feat_min = feat_min
        self.feat_max = feat_max

        print("Feature min:", self.feat_min, flush=True)
        print("Feature max:", self.feat_max, flush=True)

        print(self.path, flush=True)
        for ifeat, feature in enumerate(self.feature_names):
            print(feature, self.feat_min[ifeat], self.feat_max[ifeat], flush=True)
        
    def get_node_type(self):
        #sensor: 0
        #query: 2
        #(0, 2)--> both sensor and query, when >1 treat it as query; TODO: check the percentage of these cells
        # Get unique node_type values across all graphs.
        self.sensor_type=0
        self.query_type=2
        all_node_types = []

        for g in self.graphs:
            if hasattr(g, "node_type"):
                all_node_types.append(g.node_type.reshape(-1).cpu())

        if len(all_node_types) > 0:
            self.unique_node_types = torch.unique(torch.cat(all_node_types)).tolist()
        else:
            self.unique_node_types = []

        print("Unique graph.node_type values:", self.unique_node_types, flush=True)

    def __len__(self):
        return self.valid_length

    def __getitem__(self, index):
        if hasattr(index, '__len__') and len(index)==2:
            leadtime=index[1]
            index = index[0]
        else:
            leadtime=None  

        if self.same_time_target:
            data = self._build_same_time_sample(index)
            leadtime = 0
        else:
            if leadtime is None:
                leadtime = max(int(self.leadtime_max), 1) if self.leadtime_fixed else 1
            else:
                leadtime = max(int(leadtime), 1)
            data = self._build_forecast_sample(index, leadtime)

        data.leadtime = torch.tensor([leadtime], dtype=torch.float32).reshape(-1, 1)
        data.dataset_type = self.type
        data.graph_file = str(self.graph_file)
        data.feature_names = self.feature_names
        data.target_name = self.target_name

        #TODO: place holder for now; implement split graph for this class
        ghost_info = GhostInfo(
            #owned_mask       = owned_mask,
            ghost_rank       = torch.empty(0, dtype=torch.long),
            ghost_remote_idx = torch.empty(0, dtype=torch.long),
            local_ghost_idx  = torch.empty(0, dtype=torch.long),
            send_rank        = [],
            send_local_idx   = [],
            recv_counts      = {},
        )

        #print({"graph": data, "bcs": self.bcs, "field_labels_out": [data.target_idx]}, flush=True)
        return {"graph": data, "bcs": self.bcs, "field_labels_out": [data.target_idx.item()],
                "ghost_info": ghost_info,
                }
    def _masked_input_feature_indices(self, target_idx, num_features):
        """Return feature columns to hide on query nodes.
        For sensor-to-query regression, query nodes should not receive the value
        being predicted.  In addition to the current target column, lag-history
        columns such as ``lags_conc`` or ``lags_pm_prior`` also carry target-side
        information and are therefore masked on query nodes by default.
        """
        mask_cols = [int(target_idx)]

        if self.mask_query_lag_features:
            for i, name in enumerate(self.feature_names):
                if name not in self.query_features:
                    mask_cols.append(i)

        mask_cols = sorted(set(i for i in mask_cols if 0 <= i < num_features))
        assert len(mask_cols)==len(self.feature_names)-len(self.query_features),f"check feature_names and query_features {len(mask_cols), len(self.feature_names), len(self.query_features)}"
        return torch.tensor(mask_cols, dtype=torch.long)


    def _mask_query_input_features_(self, x, query_mask, target_idx):
        """In-place mask of target and lags* feature columns on query nodes."""
        row_idx = query_mask.view(-1).nonzero(as_tuple=True)[0]
        if row_idx.numel() == 0:
            return

        col_idx = self._masked_input_feature_indices(target_idx, x.shape[-1]).to(x.device)
        if col_idx.numel() == 0:
            return

        x[row_idx[:, None], col_idx[None, :]] = self.mask_value
    
    def _normalize_x(self, x):
        """Min-max normalize graph node features using dataset-level feature min/max."""
        feat_min = self.feat_min.to(device=x.device, dtype=x.dtype)
        feat_max = self.feat_max.to(device=x.device, dtype=x.dtype)

        denom = (feat_max - feat_min).clamp_min(1e-6)
        return 2*(x - feat_min)/denom - 1
    
    def _connect_zero_degree_nodes_to_nearest_(self, graph):
        """Connect each zero-degree node to its nearest node based on graph.pos.

        This modifies the graph in-place. For each isolated node i, it finds the
        nearest other node j using Euclidean distance in graph.pos, then adds both
        directed edges i -> j and j -> i.
        """
        num_nodes = int(graph.num_nodes)

        if num_nodes <= 1:
            return

        if not hasattr(graph, "pos") or graph.pos is None:
            raise ValueError("Cannot connect zero-degree nodes because graph.pos is missing.")

        if not hasattr(graph, "edge_index") or graph.edge_index is None:
            graph.edge_index = torch.empty((2, 0), dtype=torch.long)

        edge_index = graph.edge_index
        device = edge_index.device

        # Compute incident degree, counting both source and destination appearances.
        if edge_index.numel() == 0:
            deg = torch.zeros(num_nodes, dtype=torch.long, device=device)
        else:
            deg_src = torch.bincount(edge_index[0], minlength=num_nodes)
            deg_dst = torch.bincount(edge_index[1], minlength=num_nodes)
            deg = deg_src + deg_dst

        zero_deg_nodes = torch.where(deg == 0)[0]

        if zero_deg_nodes.numel() == 0:
            return

        pos = graph.pos.to(device=device, dtype=torch.float32)

        if not torch.isfinite(pos).all():
            raise ValueError("Cannot connect zero-degree nodes because graph.pos contains NaN/Inf.")

        # Pairwise distances from isolated nodes to all nodes.
        dist = torch.cdist(pos[zero_deg_nodes], pos)  # [num_zero, num_nodes]

        # Do not connect a node to itself.
        dist[torch.arange(zero_deg_nodes.numel(), device=device), zero_deg_nodes] = torch.inf

        nearest_nodes = dist.argmin(dim=1)
        nearest_dist = dist.min(dim=1).values

        # Add both directions: isolated -> nearest and nearest -> isolated.
        new_edges_forward = torch.stack([zero_deg_nodes, nearest_nodes], dim=0)
        new_edges_backward = torch.stack([nearest_nodes, zero_deg_nodes], dim=0)
        new_edges = torch.cat([new_edges_forward, new_edges_backward], dim=1)

        graph.edge_index = torch.cat([edge_index, new_edges], dim=1)

        # If edge_attr exists, append distance values for the new edges.
        if hasattr(graph, "edge_attr") and graph.edge_attr is not None:
            edge_attr = graph.edge_attr
            nearest_dist = nearest_dist.to(device=edge_attr.device, dtype=edge_attr.dtype)

            if edge_attr.ndim == 1:
                new_edge_attr = torch.cat([nearest_dist, nearest_dist], dim=0)
            else:
                edge_feat_dim = edge_attr.shape[1]
                new_edge_attr = nearest_dist.repeat(2).view(-1, 1)

                if edge_feat_dim > 1:
                    pad = torch.zeros(
                        new_edge_attr.shape[0],
                        edge_feat_dim - 1,
                        dtype=edge_attr.dtype,
                        device=edge_attr.device,
                    )
                    new_edge_attr = torch.cat([new_edge_attr, pad], dim=1)

            graph.edge_attr = torch.cat([edge_attr, new_edge_attr], dim=0)

        print(f"Connected {zero_deg_nodes.numel()} zero-degree nodes to nearest nodes.", flush=True)

    def _build_same_time_sample(self, index):
        raw = copy.deepcopy(self.graphs[index])

        x = self._node_features(raw)
        target_idx = self._graph_target_idx(raw)
        query_mask = self._query_mask(raw)

        # Normalize all node features before masking.
        # no need, as original data features have been scaled
        ##x = self._normalize_x(x)

        y = x[:, target_idx : target_idx + 1].clone()
        x_in = x.clone()
        if self.mask_query_target:
            self._mask_query_input_features_(x_in, query_mask, target_idx)

        raw.x = x_in.unsqueeze(1)  # [N, 1, F]
        raw.y = y
        raw.query_mask = query_mask
        raw.loss_mask = query_mask
        raw.target_idx = torch.tensor(target_idx, dtype=torch.long)
        raw.t0 = self._time_value(raw, fallback=index)
        raw.target_t = raw.t0
        raw.dt = int(self.dt)

        for name, value in {"x_in": x_in, "y": y, "edge_attr": raw.edge_attr, "pos": raw.pos}.items():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise RuntimeError(f"Non-finite {name} in sample index={index}")
        if query_mask.sum().item() == 0:
            raise RuntimeError(f"No query nodes in sample index={index}; unique node_type={torch.unique(raw.node_type)}")
        deg = torch.bincount(raw.edge_index[0], minlength=raw.num_nodes)
        zero_deg = deg == 0

        if zero_deg.any():
            zero_query = zero_deg & query_mask.view(-1).bool()
            print(f"sample index={index}: "
                f"zero-degree nodes={int(zero_deg.sum().item())}, "
                f"zero-degree query nodes={int(zero_query.sum().item())}",flush=True)
        if (deg == 0).any():
            raise ValueError(f"Warning: sample index={index} has {(deg == 0).sum().item()} zero-degree nodes")

        return raw

    def _build_forecast_sample(self, index, leadtime):
        #TODO: place holder; will need to update the function when we have an application case like this
        raise RuntimeError(f"We do not have a forecaseting case for now in {self.__class__.__name__}")
        input_graphs = [copy.deepcopy(self.graphs[index + k]) for k in range(self.nsteps_input)]
        target_index = index + self.nsteps_input + leadtime - 1
        target_graph = copy.deepcopy(self.graphs[target_index])

        for g in input_graphs[1:] + [target_graph]:
            self._assert_same_nodes(input_graphs[0], g)

        xs = []
        for g in input_graphs:
            x = self._node_features(g).clone()
            target_idx = self._graph_target_idx(g)
            query_mask = self._query_mask(g)
            if self.mask_query_target:
                x[query_mask, target_idx] = self.mask_value
            xs.append(x)

        data = input_graphs[0]
        data.x = torch.stack(xs, dim=0).permute(1, 0, 2)  # [N, n_steps, F]
        target_idx = self._graph_target_idx(target_graph)
        data.y = self._node_features(target_graph)[:, target_idx : target_idx + 1].clone()
        data.query_mask = self._query_mask(target_graph)
        data.loss_mask = data.query_mask
        data.t0 = self._time_value(input_graphs[0], fallback=index)
        data.target_t = self._time_value(target_graph, fallback=target_index)
        data.dt = int(self.dt)
        return data

    def _select_leadtime(self, index, requested_leadtime):
        if requested_leadtime is None:
            leadtime = max(int(self.leadtime_max), 1) if self.leadtime_fixed else 1
        else:
            leadtime = max(int(requested_leadtime), 1)
        max_allowed = len(self.graphs) - base_index - self.nsteps_input
        return min(leadtime, max_allowed)

    def _node_features(self, graph):
        if not hasattr(graph, "x"):
            raise AttributeError("H3 graph is missing required node feature tensor `x`.")
        x = graph.x
        if x.ndim == 3 and x.shape[1] == 1:
            x = x[:, 0, :]
        if x.ndim != 2:
            raise ValueError(f"Expected graph.x to have shape [N, F], got {tuple(x.shape)}.")
        return x.to(torch.float32)

    def _graph_target_idx(self, graph):
        idx = graph.target_idx
        if isinstance(idx, Tensor):
            idx = int(idx.reshape(-1)[0].item())
        return int(idx)

    def _query_mask(self, graph):
        if hasattr(graph, "query_mask"):
            return graph.query_mask.view(-1).to(torch.bool)
        if not hasattr(graph, "node_type"):
            # Fall back to all nodes so the dataset still works for pure graph forecasting.
            return torch.ones(graph.num_nodes, dtype=torch.bool)
        return graph.node_type.view(-1).long() >= self.query_node_min_type

    def _time_value(self, graph, fallback):
        if hasattr(graph, "time"):
            t = graph.time
            if isinstance(t, Tensor):
                return int(t.reshape(-1)[0].item())
            try:
                return int(t)
            except Exception:
                pass
        return int(fallback)

    def _assert_same_nodes(self, g0, g1):
        if g0.num_nodes != g1.num_nodes:
            raise ValueError(
                "Autoregressive H3 graph samples require stable node counts. "
                f"Got {g0.num_nodes} and {g1.num_nodes}. Use same_time_target=true for sensor-to-query regression."
            )
        if hasattr(g0, "h3_cells") and hasattr(g1, "h3_cells") and list(g0.h3_cells) != list(g1.h3_cells):
            raise ValueError(
                "Autoregressive H3 graph samples require stable H3 cell ordering. "
                "Use same_time_target=true for sensor-to-query regression."
            )

    def _load_support_options(self, supportdata):
        if supportdata is None:
            return {}
        if isinstance(supportdata, dict):
            return dict(supportdata.get(self.type, supportdata))
        if isinstance(supportdata, (str, os.PathLike)):
            path = Path(supportdata)
            if path.exists() and path.suffix.lower() in {".yaml", ".yml"}:
                if yaml is None:
                    raise RuntimeError("PyYAML is required to read supportdata YAML files.")
                with open(path, "r", encoding="utf-8") as f:
                    obj = yaml.safe_load(f) or {}
                if not isinstance(obj, dict):
                    raise ValueError(f"Expected mapping in supportdata YAML: {path}")
                return dict(obj.get(self.type, obj))
        return {}

    def _load_graph_config(self, opts):
        cfg = opts.get("config") or opts.get("graph_config") or opts.get("h3_config")
        if isinstance(cfg, dict):
            return dict(cfg)
        cfg_path = cfg or opts.get("config_path")
        candidate_paths = []
        if cfg_path:
            candidate_paths.append(Path(cfg_path))
        if os.path.isdir(self.path):
            candidate_paths += [Path(self.path) / "config.yaml", Path(self.path) / "config.yml"]
        for p in candidate_paths:
            if p.exists():
                if yaml is None:
                    raise RuntimeError("PyYAML is required to read graph config YAML files.")
                with open(p, "r", encoding="utf-8") as f:
                    obj = yaml.safe_load(f) or {}
                if not isinstance(obj, dict):
                    raise ValueError(f"Expected mapping in graph config YAML: {p}")
                return obj
        return {}

    def _resolve_graph_file(self, opts):
        explicit = opts.get("graph_file") or opts.get("pt_file")
        if explicit:
            p = Path(explicit)
            if not p.exists():
                raise FileNotFoundError(f"Configured H3 graph file does not exist: {p}")
            return str(p)

        p = Path(self.path)
        if p.is_file():
            return str(p)
        if not p.is_dir():
            raise FileNotFoundError(f"H3 graph path does not exist: {self.path}")

        patterns = []
        if self.include_string:
            patterns.append(str(p / f"*{self.include_string}*.pt"))
        patterns.extend(
            [
                str(p / f"graphs_{self.split}_*.pt"),
                str(p / f"*{self.split}*.pt"),
                str(p / "*.pt"),
            ]
        )
        matches = []
        for pat in patterns:
            matches.extend(glob.glob(pat))
        matches = sorted(set(matches))
        if not matches:
            raise FileNotFoundError(
                f"No .pt H3 graph file found under {self.path} for split={self.split!r} "
                f"and include_string={self.include_string!r}."
            )
        if len(matches) > 1 and not self.include_string:
            raise RuntimeError(
                "Multiple candidate H3 graph files found. Set include_string or supportdata.graph_file. "
                f"Candidates: {matches}"
            )
        return matches[0]

    def plot_dataset_histograms(self, save_dir="dataset_histograms", bins=50):
        """Plot and save basic dataset histograms.
        Saves:
        - num_nodes_hist.png
        - sensor_query_ratio_hist.png
        - conc_values_hist.png
        """
        import matplotlib.pyplot as plt
        from pathlib import Path

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        num_nodes_list = []
        sensor_query_ratio_list = []
        conc_values = []

        for i, graph in enumerate(self.graphs):
            x = self._node_features(graph)
            target_idx = self._graph_target_idx(graph)

            num_nodes = int(x.shape[0])
            num_nodes_list.append(num_nodes)

            if hasattr(graph, "node_type"):
                node_type = graph.node_type.view(-1).long()
                sensor_mask = node_type == 0
                query_mask = node_type >= self.query_node_min_type

                num_sensors = int(sensor_mask.sum().item())
                num_queries = int(query_mask.sum().item())

                if num_queries > 0:
                    sensor_query_ratio_list.append(num_sensors / num_queries)
                else:
                    sensor_query_ratio_list.append(float("nan"))

            conc = x[:, target_idx].detach().cpu()
            finite_mask = torch.isfinite(conc)
            if finite_mask.any():
                conc_values.append(conc[finite_mask])

        if len(conc_values) > 0:
            conc_values = torch.cat(conc_values).numpy()
        else:
            conc_values = []

        # ------------------------------------------------------------------
        # 1. Histogram: number of nodes
        # ------------------------------------------------------------------
        plt.figure()
        plt.hist(num_nodes_list, bins=bins)
        plt.xlabel("Number of nodes")
        plt.ylabel("Count")
        plt.title("Histogram of number of nodes per graph")
        plt.tight_layout()
        plt.savefig(save_dir / f"num_nodes_hist_{self.split}.png", dpi=200)
        plt.close()

        # ------------------------------------------------------------------
        # 2. Histogram: sensor/query ratio
        # ------------------------------------------------------------------
        ratio_tensor = torch.tensor(sensor_query_ratio_list, dtype=torch.float32)
        ratio_tensor = ratio_tensor[torch.isfinite(ratio_tensor)]

        if ratio_tensor.numel() > 0:
            plt.figure()
            plt.hist(ratio_tensor.numpy(), bins=bins)
            plt.xlabel("Sensor/query ratio")
            plt.ylabel("Count")
            plt.title("Histogram of sensor/query ratio per graph")
            plt.tight_layout()
            plt.savefig(save_dir / f"sensor_query_ratio_hist_{self.split}.png", dpi=200)
            plt.close()

        # ------------------------------------------------------------------
        # 3. Histogram: conc values
        # ------------------------------------------------------------------
        if len(conc_values) > 0:
            plt.figure()
            plt.hist(conc_values, bins=bins)
            plt.xlabel("conc")
            plt.ylabel("Count")
            plt.title("Histogram of conc values")
            plt.tight_layout()
            plt.savefig(save_dir / f"conc_values_hist_{self.split}.png", dpi=200)
            plt.close()

        print(f"Saved dataset histograms to: {save_dir}", flush=True)
def _write_dummy_case(out_dir):
    """Write a tiny synthetic H3 graph dataset for standalone smoke testing."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Four nodes, three input features.
    # Let feature 0 be the target field "conc".
    x0 = torch.tensor(
        [
            [1.0, 10.0, 100.0],  # sensor
            [2.0, 20.0, 200.0],  # sensor
            [3.0, 30.0, 300.0],  # query
            [4.0, 40.0, 400.0],  # query
        ],
        dtype=torch.float32,
    )

    edge_index = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 0],
        ],
        dtype=torch.long,
    )

    edge_attr = torch.ones(edge_index.shape[1], 1, dtype=torch.float32)

    graphs = []
    for t in range(3):
        g = Data(
            x=x0 + float(t),
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=torch.tensor(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                ],
                dtype=torch.float32,
            ),
            node_type=torch.tensor([0, 0, 2, 2], dtype=torch.long),
            target_idx=torch.tensor(0, dtype=torch.long),
            query_x_cols=torch.tensor([0, 1, 2], dtype=torch.long),
            time=torch.tensor(float(t), dtype=torch.float32),
            num_nodes=4,
            h3_cells=[f"dummy_cell_{i}" for i in range(4)],
        )
        graphs.append(g)

    graph_file = out_dir / "graphs_train_dummy.pt"
    torch.save(graphs, graph_file)

    config = {
        "sensor_columns": ["conc", "temperature", "humidity"],
        "query_columns": ["conc", "temperature", "humidity"],
        "target_columns": "conc",
        "position_columns": ["lat", "lon"],
        "nlags": 1,
    }

    config_file = out_dir / "config.yaml"
    if yaml is None:
        raise RuntimeError("PyYAML is required for the standalone dummy test.")
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    return graph_file, config_file

def _summarize_standalone_sample(dataset, index):
    def _shape_dtype(obj: Any) -> str:
        if torch.is_tensor(obj):
            return f"shape={tuple(obj.shape)}, dtype={obj.dtype}, device={obj.device}"
        return f"type={type(obj).__name__}"

    print("=" * 80)
    print("H3AirQualityGraphDataset standalone smoke test")
    print("=" * 80)
    print(f"dataset type: {type(dataset).__name__}")
    print(f"dataset length: {len(dataset)}")
    print(f"sample index: {index}")

    if hasattr(dataset, "graph_file"):
        print(f"graph_file: {dataset.graph_file}")
    if hasattr(dataset, "target_column"):
        print(f"target_column: {dataset.target_column}")
    if hasattr(dataset, "target_idx"):
        print(f"target_idx: {dataset.target_idx}")
    if hasattr(dataset, "sensor_columns"):
        print(f"sensor_columns: {dataset.sensor_columns}")
    if hasattr(dataset, "query_columns"):
        print(f"query_columns: {dataset.query_columns}")

    sample = dataset[index]

    print("-" * 80)
    print("returned sample")
    print("-" * 80)

    if isinstance(sample, dict):
        print(f"sample keys: {list(sample.keys())}")
        graph = sample.get("graph", None)

        for key, value in sample.items():
            if key == "graph":
                continue
            print(f"{key}: {_shape_dtype(value)}")
            if isinstance(value, (list, tuple)):
                print(f"  value: {value}")
            elif torch.is_tensor(value) and value.numel() <= 20:
                print(f"  value: {value}")

    else:
        graph = sample
        print(f"sample is not a dict: {type(sample).__name__}")

    if graph is None:
        raise RuntimeError("The dataset sample does not contain a 'graph' entry.")

    print("-" * 80)
    print("graph fields")
    print("-" * 80)

    for attr in [
        "x",
        "y",
        "edge_index",
        "edge_attr",
        "pos",
        "node_type",
        "query_x_cols",
        "target_idx",
        "time",
        "num_nodes",
    ]:
        if hasattr(graph, attr):
            value = getattr(graph, attr)
            print(f"graph.{attr}: {_shape_dtype(value)}")
            if torch.is_tensor(value) and value.numel() <= 20:
                print(f"  value: {value}")
            elif not torch.is_tensor(value):
                print(f"  value: {value}")

    if hasattr(graph, "h3_cells"):
        h3_cells = graph.h3_cells
        print(f"graph.h3_cells: len={len(h3_cells)}")
        print(f"  first few: {h3_cells[: min(5, len(h3_cells))]}")

    if hasattr(graph, "node_type"):
        node_type = graph.node_type
        if torch.is_tensor(node_type):
            query_mask = node_type >= getattr(dataset, "query_node_min_type", 1)
            print(f"num query nodes: {int(query_mask.sum().item())}")
            print(f"num total nodes: {int(node_type.numel())}")

    if hasattr(graph, "x"):
        if torch.is_tensor(graph.x):
            finite = torch.isfinite(graph.x).all().item()
            print(f"graph.x finite: {finite}")

    if hasattr(graph, "y"):
        if torch.is_tensor(graph.y):
            finite = torch.isfinite(graph.y).all().item()
            print(f"graph.y finite: {finite}")

    print("=" * 80)
    print("Smoke test passed.")
    print("=" * 80)    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=("Standalone smoke test for H3AirQualityGraphDataset. "
            "Run from the MATEY repository root as either "
            "`python matey/data_utils/h3_graph_datasets.py --dummy` or "
            "`python -m matey.data_utils.h3_graph_datasets <graph_dir_or_pt_file>`.")
    )
    parser.add_argument("path", nargs="?", default=None, help="Directory containing prebuilt .pt graphs, or a specific .pt graph file.",)
    parser.add_argument("--include_string", default="", help="Substring used to select a .pt graph file.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Dataset split name.")
    parser.add_argument("--index", type=int, default=0, help="Sample index to load and summarize.")
    parser.add_argument("--n_steps", type=int, default=1, help="Number of input steps.")
    parser.add_argument("--dt", type=int, default=1, help="Temporal stride. Currently only dt=1 is supported.")
    parser.add_argument("--config", dest="config_path", default=None, help="H3 graph-builder YAML config path.")
    parser.add_argument("--supportdata", default=None, help="Optional supportdata YAML file.")
    parser.add_argument("--graph_file", default=None, help="Explicit .pt graph file. Overrides path discovery.")
    parser.add_argument("--forecast", action="store_true", help="Test autoregressive graph-to-graph mode.")
    parser.add_argument("--no_mask_query_target", action="store_true", help="Do not mask target feature on query nodes.")
    parser.add_argument("--query_node_min_type", type=int, default=1, help="Minimum node_type considered a query node.")
    parser.add_argument("--mask_value", type=float, default=0.0, help="Value used when masking target features.")
    parser.add_argument("--dummy", action="store_true", help="Create a temporary synthetic graph file and config, then run the smoke test on it.",)
    args = parser.parse_args()
    

    tmp_ctx = tempfile.TemporaryDirectory() if args.dummy else None
    try:
        support_opts = {}
        if args.supportdata:
            support_opts.update(_load_supportdata_file(args.supportdata))

        if args.dummy:
            dummy_dir = Path(tmp_ctx.name)  # type: ignore[union-attr]
            graph_file, config_file = _write_dummy_case(dummy_dir)
            dataset_path = str(dummy_dir)
            support_opts.update({"graph_file": str(graph_file), "config_path": str(config_file)})
        else:
            dataset_path = args.path or args.graph_file
            if dataset_path is None:
                raise SystemExit("Provide a graph directory/.pt file, --graph_file, or use --dummy.")

        if args.graph_file:
            support_opts["graph_file"] = args.graph_file
        if args.config_path:
            support_opts["config_path"] = args.config_path

        support_opts["same_time_target"] = not args.forecast
        support_opts["mask_query_target"] = not args.no_mask_query_target
        support_opts["query_node_min_type"] = args.query_node_min_type
        support_opts["mask_value"] = args.mask_value

        dataset = H3AirQualityGraphDataset(
            path=dataset_path,
            include_string=args.include_string,
            n_steps=args.n_steps,
            dt=args.dt,
            supportdata=support_opts,
            split=args.split,
        )

        if not 0 <= args.index < len(dataset):
            raise IndexError(f"index must be in [0, {len(dataset)}), got {args.index}.")

        _summarize_standalone_sample(dataset, args.index)
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()