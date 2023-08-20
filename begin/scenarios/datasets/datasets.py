# COMMON
from typing import Callable, Optional
import datetime
import os
import torch
import numpy as np
import networkx as nx
import torch.nn.functional as F
import tqdm
import pickle, json
from itertools import chain

# FOR DGL DATASETS
import dgl
from dgl.data.utils import save_graphs, load_graphs, save_info, load_info, makedirs, _get_dgl_url, download, extract_archive
from ogb.graphproppred import DglGraphPropPredDataset

# for aromaticity dataset
from dgllife.data import PubChemBioAssayAromaticity
from dgllife.data.csv_dataset import MoleculeCSVDataset
from dgllife.utils.mol_to_graph import smiles_to_bigraph
import pandas as pd

class DGLGNNBenchmarkDataset(dgl.data.DGLBuiltinDataset):
    root_url = 'https://data.pyg.org/datasets/benchmarking-gnns'
    _urls = {
        'mnist': f'{root_url}/MNIST_v2.zip',
        'cifar10': f'{root_url}/CIFAR10_v2.zip',
    }
    
    def __init__(self, name, raw_dir=None, force_reload=False, verbose=True, transform=None):
        assert name.lower() in ['mnist', 'cifar10']
        url = self._urls[name]
        super(DGLGNNBenchmarkDataset, self).__init__(name, url=url, raw_dir=raw_dir, force_reload=force_reload, verbose=verbose)
        
    def convert_pyg_dict_to_dgl(self, graph_info):
        g = dgl.graph((graph_info['edge_index'][0], graph_info['edge_index'][1]), num_nodes=graph_info['x'].shape[0])
        g.ndata['feat'] = torch.cat((graph_info['x'], graph_info['pos']), dim=-1)
        g.edata['edge_attr'] = graph_info['edge_attr']
        return dgl.add_self_loop(g)
        
    def process(self):
        root = self.raw_path
        data = torch.load(os.path.join(root, f"{self.name.upper()}_v2.pt"))
        # target_idx = self.target_split_to_idx[self.target_split]
        train_graphs, train_ys = zip(*[(self.convert_pyg_dict_to_dgl(graph_info), graph_info['y']) for graph_info in tqdm.tqdm(data[0])])
        val_graphs, val_ys = zip(*[(self.convert_pyg_dict_to_dgl(graph_info), graph_info['y']) for graph_info in tqdm.tqdm(data[1])])
        test_graphs, test_ys = zip(*[(self.convert_pyg_dict_to_dgl(graph_info), graph_info['y']) for graph_info in tqdm.tqdm(data[2])])
        
        self._graphs = list(chain(train_graphs, val_graphs, test_graphs))
        train_ys, val_ys, test_ys = map(lambda x: torch.LongTensor(x), (train_ys, val_ys, test_ys))
        self.labels = torch.cat((train_ys, val_ys, test_ys), dim=0)
        self._train_masks = torch.cat((torch.ones_like(train_ys).bool(), torch.zeros_like(val_ys).bool(), torch.zeros_like(test_ys).bool()), dim=0)
        self._val_masks = torch.cat((torch.zeros_like(train_ys).bool(), torch.ones_like(val_ys).bool(), torch.zeros_like(test_ys).bool()), dim=0)
        self._test_masks = torch.cat((torch.zeros_like(train_ys).bool(), torch.zeros_like(val_ys).bool(), torch.ones_like(test_ys).bool()), dim=0)
        self._num_classes = self.labels.max().item() + 1
        self._num_feats = self._graphs[0].ndata['feat'].shape[-1]
        
    def has_cache(self):
        graph_path = os.path.join(self.save_path,
                                  self.save_name + '.bin')
        info_path = os.path.join(self.save_path,
                                 self.save_name + '.pkl')
        if os.path.exists(graph_path) and \
            os.path.exists(info_path):
            return True

        return False

    def save(self):
        """save the graph list and the labels"""
        graph_path = os.path.join(self.save_path,
                                  self.save_name + '.bin')
        info_path = os.path.join(self.save_path,
                                 self.save_name + '.pkl')
        save_graphs(str(graph_path), self._graphs, {'y': self.labels, 'train_masks': self._train_masks, 'val_masks': self._val_masks, 'test_masks': self._test_masks})
        save_info(str(info_path), {'num_classes': self._num_classes})

    def load(self):
        graph_path = os.path.join(self.save_path,
                                  self.save_name + '.bin')
        info_path = os.path.join(self.save_path,
                                 self.save_name + '.pkl')
        graphs, auxs = load_graphs(str(graph_path))
        info = load_info(str(info_path))
        self._num_classes = info['num_classes']
        
        self._graphs = graphs
        self.labels = auxs['y']
        self._train_masks = auxs['train_masks'].bool()
        self._val_masks = auxs['val_masks'].bool()
        self._test_masks = auxs['test_masks'].bool()
        self._num_feats = self._graphs[0].ndata['feat'].shape[-1]
        
        if self.verbose:
            print('num_graphs:', len(self._graphs), ', num_labels:', self.labels.shape, 'num_classes:', self._num_classes)
            
    def __getitem__(self, idx):
        # print(self._task_specific_masks.shape)
        if hasattr(self, '_task_specific_masks'):
            return self._graphs[idx], self.labels[idx], self._task_specific_masks[idx]
        else:
            return self._graphs[idx], self.labels[idx]
        
    def __len__(self):
        return len(self._graphs)

    @property
    def save_name(self):
        return self.name + '_dgl_graph'

    @property
    def num_classes(self):
        return self._num_classes
    
    @property
    def num_feats(self):
        return self._num_feats

class NYCTaxiDataset(dgl.data.DGLBuiltinDataset):
    root_url = 'https://www.dropbox.com/s/nm6w8ikxpk2x8oj'
    _urls = {
        'nyctaxi': f'{root_url}/taxi.zip?dl=1',
    }
    
    def __init__(self, name, raw_dir=None, force_reload=False, verbose=True, transform=None):
        assert name.lower() in ['nyctaxi']
        url = self._urls[name]
        super(NYCTaxiDataset, self).__init__(name, url=url, raw_dir=raw_dir, force_reload=force_reload, verbose=verbose)
    
    def process(self):
        root = self.raw_path
        days = [None, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        graphs, labels, times = [], [], []
        
        map_to_id = {'"Bronx"': 1, '"Brooklyn"': 2, '"EWR"': 3, '"Manhattan"': 4, '"Queens"': 5, '"Staten Island"': 6, '"Unknown"': 7}
        download('https://d37ci6vzurychx.cloudfront.net/misc/taxi+_zone_lookup.csv', path=os.path.join(root, 'lookup.csv'))
        with open(os.path.join(root, 'lookup.csv'), 'r') as fcsv:
            fcsv.readline()
            node_feats = torch.LongTensor([map_to_id[line.split(',')[1]] for line in fcsv])
            # print(node_feats.min(), node_feats.max(), node_feats.shape)
            node_feats = node_feats - 1
            
        for yy in tqdm.trange(2018, 2021+1):
            for mm in range(1, 12+1):
                for _dd in range(days[mm] + (1 if ((mm == 2) and (yy % 4 == 0)) else 0)):
                    dd = _dd + 1
                    srcs, dsts, edge_feats = torch.load(os.path.join(root, f"new/{yy}_{mm}_{dd}.pt"))
                    for ii in range(24):
                        labels.append(0 if (datetime.date(yy, mm, dd).weekday() < 5) else 1)
                        valid_idxs = (edge_feats[:, ii] > 0.1) | (srcs == dsts)
                        g = dgl.graph((srcs[valid_idxs], dsts[valid_idxs]))
                        g.ndata['feat'] = F.one_hot(node_feats, num_classes=7).float()
                        g.edata['feat'] = edge_feats[valid_idxs, ii:ii+1].float()
                        graphs.append(g)
                    
        self._graphs = graphs
        self.labels = torch.LongTensor(labels).unsqueeze(-1)
        # self._months = torch.LongTensor(times)
        self._num_classes = 1
        self._num_feats = self._graphs[0].ndata['feat'].shape[-1]
        
    def has_cache(self):
        graph_path = os.path.join(self.save_path,
                                  self.save_name + '.bin')
        info_path = os.path.join(self.save_path,
                                 self.save_name + '.pkl')
        if os.path.exists(graph_path) and \
            os.path.exists(info_path):
            return True

        return False

    def save(self):
        """save the graph list and the labels"""
        graph_path = os.path.join(self.save_path,
                                  self.save_name + '.bin')
        info_path = os.path.join(self.save_path,
                                 self.save_name + '.pkl')
        save_graphs(str(graph_path), self._graphs, {'y': self.labels}) # , 'mm': self._months
        save_info(str(info_path), {'num_classes': self._num_classes})

    def load(self):
        graph_path = os.path.join(self.save_path,
                                  self.save_name + '.bin')
        info_path = os.path.join(self.save_path,
                                 self.save_name + '.pkl')
        graphs, auxs = load_graphs(str(graph_path))
        info = load_info(str(info_path))
        self._num_classes = info['num_classes']
        
        self._graphs = graphs
        self.labels = auxs['y']
        # self._months = auxs['mm']
        self._num_feats = self._graphs[0].ndata['feat'].shape[-1]
        
        if self.verbose:
            print('num_graphs:', len(self._graphs), ', num_labels:', self.labels.shape, 'num_classes:', self._num_classes)
            
    def __getitem__(self, idx):
        # print(self._task_specific_masks.shape)
        if hasattr(self, '_task_specific_masks'):
            return self._graphs[idx], self.labels[idx], self._task_specific_masks[idx]
        else:
            return self._graphs[idx], self.labels[idx]
        
    def __len__(self):
        return len(self._graphs)

    @property
    def save_name(self):
        return self.name + '_dgl_graph'

    @property
    def num_classes(self):
        return self._num_classes
    
    @property
    def num_feats(self):
        return self._num_feats

    
class DglGraphPropPredDatasetWithTaskMask(DglGraphPropPredDataset):
    def __getitem__(self, idx):
        '''Get datapoint with index'''
        if hasattr(self, '_task_specific_masks'):
            if isinstance(idx, int):
                return self.graphs[idx], self.labels[idx], self._task_specific_masks[idx]
            elif torch.is_tensor(idx) and idx.dtype == torch.long:
                if idx.dim() == 0:
                    return self.graphs[idx], self.labels[idx], self._task_specific_masks[idx]
                elif idx.dim() == 1:
                    return Subset(self, idx.cpu())
        else:
            if isinstance(idx, int):
                return self.graphs[idx], self.labels[idx]
            elif torch.is_tensor(idx) and idx.dtype == torch.long:
                if idx.dim() == 0:
                    return self.graphs[idx], self.labels[idx]
                elif idx.dim() == 1:
                    return Subset(self, idx.cpu())

        raise IndexError(
            'Only integers and long are valid '
            'indices (got {}).'.format(type(idx).__name__))
        
class WikiCSLinkDataset(dgl.data.DGLBuiltinDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=False, transform=None):
        _url = _get_dgl_url('dataset/wiki_cs.zip')
        super(WikiCSLinkDataset, self).__init__(name='wiki_cs',
                                                raw_dir=raw_dir,
                                                url=_url,
                                                force_reload=force_reload,
                                                verbose=verbose)
    def process(self):
        """process raw data to graph, labels and masks"""
        with open(os.path.join(self.raw_path, 'data.json')) as f:
            data = json.load(f)
        features = torch.FloatTensor(np.array(data['features']))
        labels = torch.LongTensor(np.array(data['labels']))

        train_masks = np.array(data['train_masks'], dtype=bool).T
        val_masks = np.array(data['val_masks'], dtype=bool).T
        stopping_masks = np.array(data['stopping_masks'], dtype=bool).T
        test_mask = np.array(data['test_mask'], dtype=bool)

        edges = [[(i, j) for j in js] + [(j, i) for j in js]
                 for i, js in enumerate(data['links'])]
        edges = list(set(chain(*edges)))
        edges = torch.LongTensor([(i, j, j, i) for i, j in edges if i < j]).view(-1, 2)
        g = dgl.graph((edges[:, 0], edges[:, 1]), num_nodes = labels.shape[0])
        g.ndata['feat'] = features
        g.ndata['domain'] = labels
        self._g = g

    def has_cache(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        return os.path.exists(graph_path)

    def save(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        save_graphs(graph_path, self._g)

    def load(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        g, _ = load_graphs(graph_path)
        self._g = g[0]

    @property
    def num_classes(self):
        return 10

    def __len__(self):
        r"""The number of graphs in the dataset."""
        return 1

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self._g

class BitcoinOTCDataset(dgl.data.DGLBuiltinDataset):
    _url = 'https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz'
    _sha1_str = 'c14281f9e252de0bd0b5f1c6e2bae03123938641'

    def __init__(self, dataset_name, raw_dir=None, force_reload=False, verbose=False, transform=None):
        super(BitcoinOTCDataset, self).__init__(name='bitcoinotc',
                                                url=self._url,
                                                raw_dir=raw_dir,
                                                force_reload=force_reload,
                                                verbose=verbose)

    def download(self):
        gz_file_path = os.path.join(self.raw_dir, self.name + '.csv.gz')
        download(self.url, path=gz_file_path)
        if not dgl.data.utils.check_sha1(gz_file_path, self._sha1_str):
            raise UserWarning('File {} is downloaded but the content hash does not match.'
                              'The repo may be outdated or download may be incomplete. '
                              'Otherwise you can create an issue for it.'.format(self.name + '.csv.gz'))
        self._extract_gz(gz_file_path, self.raw_path)

    def process(self):
        filename = os.path.join(self.save_path, '../' + self.name + '.csv')
        data = np.loadtxt(filename, delimiter=',').astype(np.int64)
        data[:, 0:2] = data[:, 0:2] - data[:, 0:2].min()
        delta = datetime.timedelta(days=14).total_seconds()
        time_index = np.around((data[:, 3] - data[:, 3].min()) / delta).astype(np.int64)

        self._graphs = []
        edges = data[:, 0:2]
        rate = data[:, 2]
        # print(data[:, 0:2].min(), data[:, 0:2].max())
        g = dgl.graph((edges[:, 0], edges[:, 1]))
        g.edata['label'] = torch.LongTensor(rate.reshape(-1, 1))
        g.edata['time'] = torch.LongTensor(time_index.reshape(-1))
        g.ndata['feat'] = torch.stack((g.in_degrees(), g.out_degrees()), dim=-1).float()
        self._graphs.append(g)

    def has_cache(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        return os.path.exists(graph_path)

    def save(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        save_graphs(graph_path, self.graphs)

    def load(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        self._graphs = load_graphs(graph_path)[0]

    @property
    def graphs(self):
        return self._graphs

    def __len__(self):
        return len(self.graphs)


    def __getitem__(self, item):
        return self.graphs[item]
    
    @property
    def is_temporal(self):
        return True

    def _extract_gz(self, file, target_dir, overwrite=False):
        if os.path.exists(target_dir) and not overwrite:
            return
        print('Extracting file to {}'.format(target_dir))
        fname = os.path.basename(file)
        makedirs(target_dir)
        out_file_path = os.path.join(target_dir, fname[:-3])
        print(out_file_path)
        with gzip.open(file, 'rb') as f_in:
            with open(out_file_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

class AromaticityDataset(MoleculeCSVDataset):
    def __init__(self, raw_dir):
        self._url = 'dataset/pubchem_bioassay_aromaticity.csv'
        data_path = raw_dir + '/pubchem_bioassay_aromaticity.csv'
        download(_get_dgl_url(self._url), path=data_path, overwrite=False)
        df = pd.read_csv(data_path)

        super(AromaticityDataset, self).__init__(
            df, smiles_to_bigraph, None, None, "cano_smiles",
            './pubchem_aromaticity_dglgraph.bin', load=False, log_every=1000, n_jobs=1)
        
        label_tensor = torch.FloatTensor(self.labels).squeeze().long()
        valid_classes = torch.bincount(label_tensor) >= 20
        valid_indices = valid_classes[label_tensor].numpy().tolist()
        new_label = torch.cumsum(valid_classes.long(), dim=-1) - 1
        self.smiles = [self.smiles[i] for i in range(label_tensor.shape[0]) if valid_indices[i]]
        self.graphs = [self.graphs[i] for i in range(label_tensor.shape[0]) if valid_indices[i]]
        self.labels = torch.LongTensor([new_label[self.labels[i].long()] for i in range(label_tensor.shape[0]) if valid_indices[i]])
        self.mask   = [  self.mask[i] for i in range(label_tensor.shape[0]) if valid_indices[i]]
        for i in tqdm.trange(len(self.graphs)):
            self.graphs[i].ndata['feat'] = torch.stack((self.graphs[i].in_degrees(), self.graphs[i].out_degrees()), dim=-1).float()
        # self._labels = self.labels.clone()
        
    def __getitem__(self, idx):
        if hasattr(self, '_task_specific_masks'):
            if isinstance(idx, int):
                return self.graphs[idx], self.labels[idx], self._task_specific_masks[idx]
            elif torch.is_tensor(idx) and idx.dtype == torch.long:
                if idx.dim() == 0:
                    return self.graphs[idx], self.labels[idx], self._task_specific_masks[idx]
                elif idx.dim() == 1:
                    return Subset(self, idx.cpu())
        else:
            if isinstance(idx, int):
                return self.graphs[idx], self.labels[idx]
            elif torch.is_tensor(idx) and idx.dtype == torch.long:
                if idx.dim() == 0:
                    return self.graphs[idx], self.labels[idx]
                elif idx.dim() == 1:
                    return Subset(self, idx.cpu())
                
class TwitchGamerNodeDataset(dgl.data.DGLBuiltinDataset):
    _url = 'http://snap.stanford.edu/data/twitch_gamers.zip'

    def __init__(self, dataset_name, raw_dir=None, force_reload=False, verbose=False, transform=None):
        super(TwitchGamerNodeDataset, self).__init__(name='twitch',
                                                 url=self._url,
                                                 raw_dir=raw_dir,
                                                 force_reload=force_reload,
                                                 verbose=verbose)
    def process(self):
        self._graphs = []
        edgefile = os.path.join(self.save_path, 'large_twitch_edges.csv')
        edgedata = pd.read_csv(edgefile)
        nodefile = os.path.join(self.save_path, 'large_twitch_features.csv')
        nodedata = pd.read_csv(nodefile)
        lang_to_domain = {'CS': 0, 'DA': 1, 'DE': 2, 'EN': 3, 'ES': 4, 'FI': 5, 'FR': 6, 'HU': 7, 'IT': 8, 'JA': 9,
                          'KO': 10, 'NL': 11, 'NO': 12, 'PL': 13, 'PT': 14, 'RU': 15, 'SV': 16, 'TH': 17, 'TR': 18, 'ZH': 19,
                          'OTHER': 20}
        graph = dgl.graph((edgedata['numeric_id_1'].values, edgedata['numeric_id_2'].values))

        langs = nodedata['language'].values.tolist()
        graph.ndata['domain'] = torch.LongTensor([lang_to_domain[_lang] for _lang in langs])
        normalized_views = torch.FloatTensor(nodedata['views'].values / float(nodedata['views'].values.max()))
        matures = torch.FloatTensor(nodedata['mature'].values)
        lifetimes = torch.FloatTensor(nodedata['life_time'].values)
        is_deads = torch.FloatTensor(nodedata['dead_account'].values)
        graph.ndata['feat'] = torch.stack((normalized_views, matures, lifetimes, is_deads), dim=-1)
        graph.ndata['label'] = torch.LongTensor(nodedata['affiliate'].values)
        self._graphs.append(graph)

    def has_cache(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        return os.path.exists(graph_path)

    def save(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        save_graphs(graph_path, self.graphs)

    def load(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        self._graphs = load_graphs(graph_path)[0]

    @property
    def graphs(self):
        return self._graphs

    @property
    def num_classes(self):
        return 2

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, item):
        return self.graphs[item]
    
class OgbgPpaSampledDataset:
    def __init__(self, save_path):
        dataset = DglGraphPropPredDataset(name = 'ogbg-ppa', root=save_path)
        pkl_path = os.path.join(save_path, f'ogbg-ppa_metadata_domainIL.pkl')
        download(f'https://github.com/jihoon-ko/BeGin/raw/main/metadata/ogbg-ppa_metadata_domainIL.pkl', path=pkl_path)
        metadata = pickle.load(open(pkl_path, 'rb'))
        self.graphs = []
        for i in tqdm.tqdm(metadata['sampled_indices']):
            graph = dataset.graphs[i]
            graph.ndata['feat'] = torch.stack((graph.in_degrees(), graph.out_degrees()), dim=-1).float()
            self.graphs.append(graph)
            
        self.labels = dataset.labels[metadata['sampled_indices']].squeeze()
        
    def __getitem__(self, idx):
        '''Get datapoint with index'''
        if isinstance(idx, int):
            return self.graphs[idx], self.labels[idx]
        elif torch.is_tensor(idx) and idx.dtype == torch.long:
            if idx.dim() == 0:
                return self.graphs[idx], self.labels[idx]
            elif idx.dim() == 1:
                return Subset(self, idx.cpu())

        raise IndexError(
            'Only integers and long are valid '
            'indices (got {}).'.format(type(idx).__name__))
    
    def __len__(self):
        return len(self.graphs)
    
class AskUbuntuDataset(dgl.data.DGLBuiltinDataset):
    _url = 'http://snap.stanford.edu/data/sx-askubuntu.txt.gz'
    
    def __init__(self, dataset_name, raw_dir=None, force_reload=False, verbose=False, transform=None):
        super(AskUbuntuDataset, self).__init__(name='askubuntu',
                                                url=self._url,
                                                raw_dir=raw_dir,
                                                force_reload=force_reload,
                                                verbose=verbose)

    def download(self):
        gz_file_path = os.path.join(self.raw_dir, self.name + '.txt.gz')
        download(self.url, path=gz_file_path)
        self._extract_gz(gz_file_path, self.raw_path)


    def process(self):
        filename = os.path.join(self.save_path, '../askubuntu.txt')
        data = np.loadtxt(filename, delimiter=' ').astype(np.int64)
        srcs, dsts, timestamps = data[:, 0].tolist(), data[:, 1].tolist(), data[:, 2].tolist()
        
        edges = set([])
        uniques = []
        interactions = zip(srcs, dsts, timestamps)
        interactions = sorted(interactions, key=lambda x: x[2])
        for s, d, t in interactions:
            if (s, d) not in edges and (d, s) not in edges:
                edges.add((s, d))
                edges.add((d, s))
                uniques.append((s, d, t))
        
        srcs, dsts, timestamps = zip(*uniques)
        months = []
        for t in timestamps:
            dt = datetime.datetime.utcfromtimestamp(t)
            months.append(dt.year * 12 + dt.month)

        months = torch.LongTensor(months)
        srcs, dsts = torch.LongTensor(srcs), torch.LongTensor(dsts)
        months = months - months.min()
        
        chosen_indices = (months >= 18)
        srcs, dsts, months = srcs[chosen_indices], dsts[chosen_indices], months[chosen_indices]
        months = months - months.min()
        print(srcs.shape, months.shape, months.max())
        bi_srcs = torch.stack((srcs, dsts), dim=-1).view(-1)
        bi_dsts = torch.stack((dsts, srcs), dim=-1).view(-1)
        bi_timestamps = torch.repeat_interleave(months, 2, dim=0)
        
        self._graphs = []
        g = dgl.graph((bi_srcs, bi_dsts))
        g.edata['time'] = bi_timestamps
        g.ndata['feat'] = g.in_degrees().float().unsqueeze(-1)
        self._graphs.append(g)

    def has_cache(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        return os.path.exists(graph_path)

    def save(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        save_graphs(graph_path, self._graphs)

    def load(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        self._graphs = load_graphs(graph_path)[0]

    @property
    def graphs(self):
        return self._graphs

    def __len__(self):
        return len(self._graphs)


    def __getitem__(self, item):
        return self._graphs[item]
    
    @property
    def is_temporal(self):
        return True

    def _extract_gz(self, file, target_dir, overwrite=False):
        if os.path.exists(target_dir) and not overwrite:
            return
        print('Extracting file to {}'.format(target_dir))
        fname = os.path.basename(file)
        makedirs(target_dir)
        out_file_path = os.path.join(target_dir, fname[:-3])
        print(out_file_path)
        with gzip.open(file, 'rb') as f_in:
            with open(out_file_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print("DONE")
        
class FacebookLinkDataset(dgl.data.DGLBuiltinDataset):
    _url = 'http://snap.stanford.edu/data/gemsec_facebook_dataset.tar.gz'
    
    def __init__(self, dataset_name, raw_dir=None, force_reload=False, verbose=False, transform=None):
        super(FacebookLinkDataset, self).__init__(name='facebook',
                                                  url=self._url,
                                                  raw_dir=raw_dir,
                                                  force_reload=force_reload,
                                                  verbose=verbose)

    def download(self):
        gz_file_path = os.path.join(self.raw_dir, self.name + '.tar.gz')
        download(self.url, path=gz_file_path)
        extract_archive(gz_file_path, self.raw_path)

    def process(self):
        domains = ['artist', 'athletes', 'company', 'government', 'new_sites', 'politician', 'public_figure', 'tvshow']
        node_cnt = 0
        all_srcs, all_dsts, all_domains = [], [], []
        # all_neg_edges = []
        for i, domain in enumerate(domains):
            filename = os.path.join(self.save_path, 'facebook_clean_data/' + domain + '_edges.csv')
            y = pd.read_csv(filename)
            srcs, dsts = y['node_1'].values, y['node_2'].values
            num_nodes = max(srcs.max(), dsts.max())
            """
            edges = set(zip(srcs, dsts)).union(set(zip(dsts, srcs)))
            neg_edges = []
            while len(neg_edges) < 25000:
                s, d = np.random.randint(num_nodes), np.random.randint(num_nodes)
                if s >= d: continue
                if (s, d) not in edges:
                    neg_edges.append((s, d))
            all_neg_edges.append(torch.LongTensor(neg_edges) + node_cnt)
            """
            all_srcs.append(torch.LongTensor(srcs) + node_cnt)
            all_dsts.append(torch.LongTensor(dsts) + node_cnt)
            all_domains.append(i * torch.ones(srcs.shape[0], dtype=torch.long))
            node_cnt += num_nodes
        
        all_srcs, all_dsts, all_domains = torch.cat(all_srcs), torch.cat(all_dsts), torch.cat(all_domains)
        bi_srcs = torch.stack((all_srcs, all_dsts), dim=-1).view(-1)
        bi_dsts = torch.stack((all_dsts, all_srcs), dim=-1).view(-1)
        bi_domains = torch.repeat_interleave(all_domains, 2, dim=0)
        self._graphs = []
        graph = dgl.graph((bi_srcs, bi_dsts))
        graph.edata['domain'] = bi_domains
        graph.ndata['feat'] = graph.in_degrees().float().unsqueeze(-1)
        
        """
        all_neg_edges = torch.cat(all_neg_edges, dim=0)
        metadata = {}
        metadata['inner_tvt_splits'] = torch.randperm(all_domains.shape[0]) % 10
        metadata['neg_edges'] = {'val': all_neg_edges[0::2], 'test': all_neg_edges[1::2]}
        pickle.dump(metadata, open('facebook_metadata_domainIL.pkl', 'wb'))
        """
        self._graphs.append(graph)
        
    def has_cache(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        return os.path.exists(graph_path)

    def save(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        save_graphs(graph_path, self._graphs)

    def load(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        self._graphs = load_graphs(graph_path)[0]

    @property
    def graphs(self):
        return self._graphs

    def __len__(self):
        return len(self._graphs)


    def __getitem__(self, item):
        return self._graphs[item]
    
    @property
    def is_temporal(self):
        return True

class SentimentGraphDataset(dgl.data.DGLBuiltinDataset):
    _url = 'https://github.com/jihoon-ko/BeGin/raw/main/metadata/sentiment_metadata_allIL.pkl'
    
    def __init__(self, dataset_name, raw_dir=None, force_reload=False, verbose=False, transform=None):
        super(SentimentGraphDataset, self).__init__(name='sentiment',
                                                  url=self._url,
                                                  raw_dir=raw_dir,
                                                  force_reload=force_reload,
                                                  verbose=verbose)

    def download(self):
        pkl_file_path = os.path.join(self.save_path, self.name + '_metadata_allIL.pkl')
        download(self.url, path=pkl_file_path)
        
    def process(self):
        metadata = pickle.load(open(os.path.join(self.save_path, 'sentiment_metadata_allIL.pkl'), 'rb'))
        # {'graphs': graphs, 'inner_tvt_splits': inner_tvt_splits, 'time_info': time_info}
        
        self._graphs = []
        self.labels = []
        for g_data in metadata['graphs']:
            # {'idx': idx, 'nfeats': torch.FloatTensor(nfeats), 'srcs': torch.LongTensor(srcs), 'dsts': torch.LongTensor(dsts), 'labels': rats[idx]}
            nfeats, srcs, dsts, label = g_data['nfeats'], g_data['srcs'], g_data['dsts'], g_data['labels']
            g = dgl.graph((srcs, dsts), num_nodes=nfeats.shape[0])
            g.ndata['feat'] = torch.FloatTensor(nfeats)
            self._graphs.append(dgl.add_self_loop(g))
            self.labels.append(1 if label > 0 else 0)
            
        self.labels = torch.LongTensor(self.labels)
        print(torch.bincount(self.labels))
        
        inner_tvt_splits = metadata['inner_tvt_splits'] 
        self._train_masks = (inner_tvt_splits % 10) < 6
        self._val_masks = ((inner_tvt_splits % 10) == 6) | ((inner_tvt_splits % 10) == 7)
        self._test_masks = (inner_tvt_splits % 10) > 7
        self._num_classes = self.labels.max().item() + 1
        self._num_feats = self._graphs[0].ndata['feat'].shape[-1]
        self._time_info = metadata['time_info']
        
    def has_cache(self):
        graph_path = os.path.join(self.save_path,
                                  self.save_name + '.bin')
        info_path = os.path.join(self.save_path,
                                 self.save_name + '.pkl')
        if os.path.exists(graph_path) and \
            os.path.exists(info_path):
            return True

        return False

    def save(self):
        """save the graph list and the labels"""
        graph_path = os.path.join(self.save_path,
                                  self.save_name + '.bin')
        info_path = os.path.join(self.save_path,
                                 self.save_name + '.pkl')
        save_graphs(str(graph_path), self._graphs, {'y': self.labels, 'train_masks': self._train_masks, 'val_masks': self._val_masks, 'test_masks': self._test_masks, 'time_info': self._time_info})
        save_info(str(info_path), {'num_classes': self._num_classes})

    def load(self):
        graph_path = os.path.join(self.save_path,
                                  self.save_name + '.bin')
        info_path = os.path.join(self.save_path,
                                 self.save_name + '.pkl')
        graphs, auxs = load_graphs(str(graph_path))
        info = load_info(str(info_path))
        self._num_classes = info['num_classes']
        
        self._graphs = graphs
        self.labels = auxs['y']
        self._train_masks = auxs['train_masks'].bool()
        self._val_masks = auxs['val_masks'].bool()
        self._test_masks = auxs['test_masks'].bool()
        self._time_info = auxs['time_info'].long()
        self._num_feats = self._graphs[0].ndata['feat'].shape[-1]
        
        if self.verbose:
            print('num_graphs:', len(self._graphs), ', num_labels:', self.labels.shape, 'num_classes:', self._num_classes)
            
    def __getitem__(self, idx):
        # print(self._task_specific_masks.shape)
        if hasattr(self, '_task_specific_masks'):
            return self._graphs[idx], self.labels[idx], self._task_specific_masks[idx]
        else:
            return self._graphs[idx], self.labels[idx]
        
    def __len__(self):
        return len(self._graphs)

    @property
    def save_name(self):
        return self.name + '_dgl_graph'

    @property
    def num_classes(self):
        return self._num_classes
    
    @property
    def num_feats(self):
        return self._num_feats