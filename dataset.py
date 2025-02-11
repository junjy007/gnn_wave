import netCDF4 as nc
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import torch_geometric
from sklearn.preprocessing import MinMaxScaler
from config import Config
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import pickle

def treat_angle(th):
    th1 = th.copy() - 180
    return th1

def load_wave_data(config:Config):
    data_fullpath = os.path.join(config.data_dir, config.data_fname)
    dataset = nc.Dataset(data_fullpath)
    # Extract variables
    data = {}
    data['x'] = dataset.variables['x'][:]
    data['y'] = dataset.variables['y'][:]
    data['lon'] = dataset.variables['lon'][:]
    data['lat'] = dataset.variables['lat'][:]
    data['nv'] = dataset.variables['nv'][:]
    data['t'] = dataset.variables['Itime'][:]
    data['t2'] = dataset.variables['Itime2'][:]
    data['h'] = dataset.variables['h'][:]
    data['hs'] = dataset.variables['hs'][:]
    data['ub_bot'] = dataset.variables['ub_bot'][:]
    data['tpeak'] = dataset.variables['tpeak'][:]
    data['wlen'] = dataset.variables['wlen'][:]
    data['pwave_bot'] = dataset.variables['pwave_bot'][:]
    # Adjust 'dirm' data if it exists
    if 'dirm' in dataset.variables:
        data['dirm'] = treat_angle(dataset.variables['dirm'][:])  # Adjust the angle data
    return data

def normalise_data(data):
    keys = list(data.keys())
    normalizers = {} 
    for k in keys:
        assert k+"_n" not in data.keys(), "already normalised"
        if k not in ['hs', 'ub_bot', 'wlen', 'pwave_bot', 'tpeak', 'dirm']:
            continue
        # Normalize the variables
        scaler = MinMaxScaler()
        data[k+"_n"] = scaler.fit_transform(data[k])
        # Each feature is in 0-1, i.e. if the variable is [T, D]
        # Normalisation is for each T-dimensional record of one feature
        # is normalised to 0-1
        data[k+"_normalizer"] = scaler
        normalizers[k + "_normalizer"] = scaler
    
    # Save the dictionary of fitted scalers
    with open('normalizers_v3.pkl', 'wb') as f:
        pickle.dump(normalizers, f)

    return data

################################################################################
def build_torch_graph(data, config:Config):
    nnsize = config.neighbourhood_size

    # Create a NearestNeighbors instance
    lon = data['lon']
    lat = data['lat']
    nbrs = NearestNeighbors(n_neighbors=nnsize+1, algorithm='ball_tree')\
        .fit(np.vstack((lon, lat)).T)

    # Get the indices and distances to the nearest neighbors for each point
    distances, indices = nbrs.kneighbors(np.vstack((lon, lat)).T)

    # Create edge index for PyTorch Geometric graph
    edge_index = []
    for i in range(indices.shape[0]):
        for j in range(1, indices.shape[1]):
            edge_index.append((i, indices[i, j]))
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return edge_index

def get_torch_graph_data(t, data, config:Config, feature_set):

    # Define features and labels
    # Here x can be a combination of your different data features at a specific time slice
    #x = torch.tensor(np.array([hs[0,:], ub_bot[0,:], dirm[0,:]]).T, dtype=torch.float) #previous one 3 features
    normalize_suffix = "_n" if config.normalization != "none" else ""
    if feature_set.lower() in ["observe", "x"]:
        variable_names = config.node_var_observ
    elif feature_set.lower() in ["target", "y"]:
        variable_names = config.node_var_target
    data_array = [
        data[k + normalize_suffix][t, :] for k in variable_names
    ]
    x = torch.tensor(np.array(data_array), dtype=torch.float32).T
    return x


class MyGraphDataset:
    def __init__(self, raw_data, config:Config, mode="train"):
        # REFERENCEs
        self.raw_data = raw_data
        self.config = config
        self.graph_edge_index = build_torch_graph(raw_data, self.config) # build the graph for future usage
        if mode == "train":
            self.t0 = config.train_period[0]
            self.t1 = config.train_period[1]
        elif mode == "test":
            self.t0 = config.test_period[0]
            self.t1 = config.test_period[1]
        else:
            raise ValueError("model must be [train|test]")

    def __getitem__(self, t):
        ss = self.config.snapshots
        t = self.t0 + t #+ ss - 3

        # Stack a few snapshots into observations in one sample
        tmp_x_array = []
        for ti in range(t, t-ss, -1):
            # Creating PyTorch Geometric graph data
            x = get_torch_graph_data(ti, self.raw_data, self.config, "observe")
            tmp_x_array.append(x)
        x = torch.concat(tmp_x_array, dim=-1)

        # Organize sample into a format recognized by torch geometry
        # - we need to trim y according to self.config.prediction_range
        x = torch_geometric.data.Data(x=x, edge_index=self.graph_edge_index)
        y = get_torch_graph_data(t + self.config.forward_time, self.raw_data, self.config, "target")
        return x, y

    def __len__(self):
        return self.t1 - self.t0 #- (self.config.snapshots-3)

def worker_init_fn(worker_id, g):
    # Each worker will have different seed (worker_id is different for each worker)
    g.manual_seed(g.initial_seed() + worker_id)

def get_data_loader(raw_data, mode, config:Config):
    # Initialize a generator and set a seed for it
    worker_init_fn = None
    if mode == "train":
        dataset = MyGraphDataset(raw_data, config, mode="train")
        batch_size = config.train_batch_size
        shuffle = config.train_shuffle
        if shuffle:
            g = torch.Generator()
            g.manual_seed(config.data_shuffle_seed)
            worker_init_fn = lambda i:worker_init_fn(i, g)
    elif mode == "test":
        dataset = MyGraphDataset(raw_data, config, mode="test")
        batch_size = config.test_batch_size
        shuffle = False
    else:
        raise ValueError("mode must be either 'train' or 'test'")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, worker_init_fn=worker_init_fn)
    return loader

def prepare_train_test_dataloaders(config:Config):
    # High level function to setup the data loaders for an experiment
    data = load_wave_data(config)
    if config.normalization == "norm_01":
        raw_data = normalise_data(data)
    else:
        raise ValueError("Normalisation must be norm_01 for now")
    train_dl = get_data_loader(raw_data, "train", config)
    test_dl = get_data_loader(raw_data, "test", config)
    return train_dl, test_dl


def unbatch(pred, batch):
    max_batch = batch.max().item() + 1
    return pred.split(batch.bincount().tolist())