import dgl
import torch
from ..datasets.datasets import DGLGalkeLifelongDataset, WikiCSLinkDataset, DGLGNNBenchmarkDataset, NYCTaxiDataset
from ogb.nodeproppred import DglNodePropPredDataset
from ogb.linkproppred import DglLinkPropPredDataset
from ..datasets.datasets import DglGraphPropPredDatasetWithTaskMask as DglGraphPropPredDataset

from .evaluator import *
evaluator_map = {'accuracy': AccuracyEvaluator, 'rocauc': ROCAUCEvaluator, 'hits': HitsEvaluator}

import os, pickle
from dgl.data.utils import download, Subset

def load_node_dataset(dataset_name, save_path):
    cover_rule = {'feat': 'node', 'label': 'node', 'train_mask': 'node', 'val_mask': 'node', 'test_mask': 'node'}
    if dataset_name in ['cora']:
        dataset = dgl.data.CoraGraphDataset(raw_dir=save_path, verbose=True)
        graph = dataset._g
        num_feats, num_classes = graph.ndata['feat'].shape[-1], dataset.num_classes
    elif dataset_name in ['citeseer']:
        dataset = dgl.data.CiteseerGraphDataset(raw_dir=save_path)
        graph = dataset._g
        num_feats, num_classes = graph.ndata['feat'].shape[-1], dataset.num_classes
    elif dataset_name in ['dblp-easy', 'dblp-hard', 'pharmabio']:
        dataset = DGLGalkeLifelongDataset(dataset_name, raw_dir=save_path)
        graph = dataset._g
        num_feats, num_classes = graph.ndata['feat'].shape[-1], dataset.num_classes
    elif dataset_name in ['ogbn-products', 'ogbn-arxiv']:
        dataset = DglNodePropPredDataset(dataset_name, root=save_path)
        split_idx = dataset.get_idx_split()
        graph, label = dataset[0]
        srcs, dsts = graph.all_edges()
        graph.add_edges(dsts, srcs)
        num_feats, num_classes = graph.ndata['feat'].shape[-1], dataset.num_classes
        for _split, _split_name in [('train', 'train'), ('valid', 'val'), ('test', 'test')]:
            _indices = torch.zeros(graph.num_nodes(), dtype=torch.bool)
            _indices[split_idx[_split]] = True
            graph.ndata[_split_name + '_mask'] = _indices
        graph.ndata['label'] = label.squeeze()
        if dataset_name == 'ogbn-arxiv':
            graph.ndata['time'] = graph.ndata.pop('year').squeeze()
    elif dataset_name in ['ogbn-proteins']:
        dataset = DglNodePropPredDataset(dataset_name, root=save_path)
        graph, label = dataset[0]
        # print(label.min(0).values, label.max(0).values)
        graph.ndata['feat'] = scatter(graph.edata.pop('feat'), graph.edges()[0], dim=0, reduce='mean')
        unique_ids = torch.unique(graph.ndata['species'])
        raw_species_to_domain = -torch.ones(unique_ids.max().item() + 1, dtype=torch.long)
        raw_species_to_domain[unique_ids] = torch.arange(8)
        graph.ndata['species'] = raw_species_to_domain[graph.ndata.pop('species').squeeze(-1)]
        print(torch.bincount(graph.ndata['species']))
        
        split_idx = dataset.get_idx_split()
        num_feats, num_classes = graph.ndata['feat'].shape[-1], label.shape[-1]
        for _split, _split_name in [('train', 'train'), ('valid', 'val'), ('test', 'test')]:
            _indices = torch.zeros(graph.num_nodes(), dtype=torch.bool)
            _indices[split_idx[_split]] = True
            graph.ndata[_split_name + '_mask'] = _indices
        graph.ndata['label'] = label
        graph.ndata['domain'] = graph.ndata.pop('species').squeeze()
    
    for k in graph.ndata.keys():
        if k not in cover_rule:
            cover_rule[k] = 'node'
    for k in graph.edata.keys():
        if k not in cover_rule:
            cover_rule[k] = 'edge'
    
    srcs, dsts = graph.edges()
    is_non_loop = (srcs != dsts)
    final_graph = dgl.graph((srcs[is_non_loop], dsts[is_non_loop]), num_nodes=graph.num_nodes())
    for k in graph.ndata.keys():
        final_graph.ndata[k] = graph.ndata[k]
    for k in graph.edata.keys():
        final_graph.edata[k] = graph.edata[k][is_non_loop]
        
    final_graph = dgl.add_self_loop(final_graph)
    return num_classes, num_feats, final_graph, cover_rule

def load_link_dataset(dataset_name, save_path):
    if dataset_name == 'ogbl-collab':
        dataset = DglLinkPropPredDataset(dataset_name, root=save_path)
        split_edge = dataset.get_edge_split()
        train_graph = dataset[0]
        combined = {}
        for k in split_edge["train"].keys():
            combined[k] = torch.cat((split_edge["train"][k], split_edge["valid"][k], split_edge["test"][k]), dim=0)
            original = combined[k]
            if k == 'edge':
                rev_edges = torch.cat((combined['edge'][:, 1:2], combined['edge'][:, 0:1]), dim=-1)
                combined[k] = torch.cat((combined[k], rev_edges), dim=-1).view(-1, 2)
            else:
                combined[k] = torch.repeat_interleave(combined[k], 2, dim=0)
        graph = dgl.graph((combined['edge'][:, 0], combined['edge'][:, 1]), num_nodes=train_graph.num_nodes())
        for k in combined.keys():
            if k != 'edge':
                if k == 'year': graph.edata['time'] = combined[k]
                else: graph.edata[k] = combined[k]
        for k in train_graph.ndata.keys():
            graph.ndata[k] = train_graph.ndata[k]
        
        _srcs, _dsts = map(lambda x: x.numpy().tolist(), graph.edges())
        edgeset = {(s, d) for s, d in zip(_srcs, _dsts)}
        neg_edges = {}
        neg_edges['val'] = torch.LongTensor([[_s, _d] for _s, _d in zip(*zip(*split_edge['valid']['edge_neg'].numpy().tolist())) if (_s, _d) not in edgeset])
        neg_edges['test'] = torch.LongTensor([[_s, _d] for _s, _d in zip(*zip(*split_edge['test']['edge_neg'].numpy().tolist())) if (_s, _d) not in edgeset])
    elif dataset_name == 'wikics':
        dataset = WikiCSLinkDataset(raw_dir=save_path)
        graph = dataset._g
        neg_edges = {}
    elif dataset_name == 'ogbl-ppa':
        dataset = DglLinkPropPredDataset(dataset_name, root=save_path)
        split_edge = dataset.get_edge_split()
        graph = dataset[0]
        graph.ndata['domain'] = graph.ndata['feat'].argmax(-1)
        neg_edges = {}
        neg_edges['val_pos'] = split_edge['valid']['edge']
        neg_edges['test_pos'] = split_edge['test']['edge']
        neg_edges['val'] = split_edge['valid']['edge_neg']
        neg_edges['test'] = split_edge['test']['edge_neg']
    
    return graph.ndata['feat'].shape[-1], graph, neg_edges

def load_graph_dataset(dataset_name, save_path):
    if dataset_name in ['mnist', 'cifar10']:
        dataset = DGLGNNBenchmarkDataset(dataset_name, raw_dir=save_path)
        num_feats, num_classes = dataset.num_feats, dataset.num_classes
    elif dataset_name in ['ogbg-molhiv']:
        dataset = DglGraphPropPredDataset(dataset_name, root=save_path)
        num_feats, num_classes = dataset[0][0].ndata['feat'].shape[-1], 1
        split_idx = dataset.get_idx_split()
        for _split, _split_name in [('train', '_train'), ('valid', '_val'), ('test', '_test')]:
            _indices = torch.zeros(len(dataset), dtype=torch.bool)
            _indices[split_idx[_split]] = True
            setattr(dataset, _split_name + '_mask', _indices)
        # dataset._labels = dataset.labels.squeeze()
    elif dataset_name in ['nyctaxi']:
        dataset = NYCTaxiDataset(dataset_name, raw_dir=save_path)
        num_feats, num_classes = dataset[0][0].ndata['feat'].shape[-1], 2
        
    return num_classes, num_feats, dataset

class DGLBasicIL:
    """두 개의 int 값을 입력받아 다양한 연산을 할 수 있도록 하는 클래스.

    :param int a: a 값
    :param int b: b 값
    """
    def __init__(self, dataset_name=None, save_path='/mnt/d/graph_dataset', num_tasks=1, incr_type='class', cover_unseen=True, minimize=True, metric=None, **kwargs):
        """미리 입력받은 a와 b값이 같은지 확인하여 결과를 반환합니다.

        :return: boolean True or False에 대한 결과, a와 b가 값으면 True, 다르면 False

        예제:
            다음과 같이 사용하세요:

            >>> Test(1, 2).is_same()
            False

        """
        self.dataset_name = dataset_name
        self.save_path = save_path
        self.num_classes = None
        self.num_feats = None
        self.num_tasks = num_tasks
        self.incr_type = incr_type
        self.cover_unseen = cover_unseen
        self.minimize = minimize
        self.metric = metric
        self.kwargs = kwargs
        
        self._curr_task = 0
        self._target_dataset = None
        
        self._init_continual_scenario()
        
        self._update_target_dataset()
        self._update_accumulated_dataset()
        
    def _init_continual_scenario(self):
        raise NotImplementedError
    
    def _update_target_dataset(self):
        raise NotImplementedError
    
    def _update_accumulated_dataset(self):
        raise NotImplementedError
    
    def __len__(self):
        return self.num_tasks
    
    def next_task(self, preds=torch.empty(1)):
        self._curr_task += 1
        if self._curr_task < self.num_tasks:
            self._update_target_dataset()
            self._update_accumulated_dataset()
            
    def get_current_dataset(self):
        if self._curr_task >= self.num_tasks: return None
        return self._target_dataset
    
    def get_accumulated_dataset(self):
        if self._curr_task >= self.num_tasks: return None
        return self._accumulated_dataset

class DGLNodeClassificationIL(DGLBasicIL):
    def _init_continual_scenario(self):
        self.num_classes, self.num_feats, self.__graph, self.__cover_rule = load_node_dataset(self.dataset_name, self.save_path)
        
        if self.incr_type in ['class', 'task']:
            if self.kwargs is not None and 'task_orders' in self.kwargs:
                self.__splits = tuple([torch.LongTensor(class_ids) for class_ids in self.kwargs['task_orders']])
            elif self.kwargs is not None and 'task_shuffle' in self.kwargs and self.kwargs['task_shuffle']:
                self.__splits = torch.split(torch.randperm(self.num_classes), self.num_classes // self.num_tasks)[:self.num_tasks]
            else:
                self.__splits = torch.split(torch.arange(self.num_classes), self.num_classes // self.num_tasks)[:self.num_tasks]
            print('class split information:', self.__splits)
            id_to_task = self.num_tasks * torch.ones(self.num_classes).long()
            for i in range(self.num_tasks):
                id_to_task[self.__splits[i]] = i
            self.__task_ids = id_to_task[self.__graph.ndata['label']]
            self.__graph.ndata['test_mask'] = self.__graph.ndata['test_mask'] & (self.__task_ids < self.num_tasks)
        elif self.incr_type == 'time':
            pkl_path = os.path.join(self.save_path, f'{self.dataset_name}_metadata_timeIL.pkl')
            download(f'https://github.com/anonymous-submission-23/anonymous-submission-23.github.io/raw/main/_splits/{self.dataset_name}_metadata_timeIL.pkl', pkl_path)
            metadata = pickle.load(open(pkl_path, 'rb'))
            inner_tvt_splits = metadata['inner_tvt_splits']
            self.__time_splits = metadata['time_splits']
            
            # split train:val:test = 4:1:5
            self.__graph.ndata['train_mask'] = (inner_tvt_splits < 4)
            self.__graph.ndata['val_mask'] = (inner_tvt_splits == 4)
            self.__graph.ndata['test_mask'] = (inner_tvt_splits > 4)
            self.num_tasks = len(self.__time_splits) - 1
            self.__task_ids = torch.zeros_like(self.__graph.ndata['time'])
            for i in range(1, self.num_tasks):
                self.__task_ids[self.__graph.ndata['time'] >= self.__time_splits[i]] = i
        elif self.incr_type == 'domain':
            self.num_tasks = self.__graph.ndata['domain'].max().item() + 1
            if self.kwargs is not None and 'task_shuffle' in self.kwargs and self.kwargs['task_shuffle']:
                self.__task_order = torch.randperm(self.num_tasks)
                print('domain_order:', self.__task_order)
                self.__task_ids = self.__task_order[self.__graph.ndata['domain']]
            else:
                self.__task_ids = self.__graph.ndata['domain']
            
            pkl_path = os.path.join(self.save_path, f'{self.dataset_name}_metadata_domainIL.pkl')
            download(f'https://github.com/anonymous-submission-23/anonymous-submission-23.github.io/raw/main/_splits/{self.dataset_name}_metadata_domainIL.pkl', pkl_path)
            metadata = pickle.load(open(pkl_path, 'rb'))
            inner_tvt_splits = metadata['inner_tvt_splits']
            self.__graph.ndata['train_mask'] = (inner_tvt_splits < 4)
            self.__graph.ndata['val_mask'] = (inner_tvt_splits == 4)
            self.__graph.ndata['test_mask'] = (inner_tvt_splits > 4)
            
        if self.incr_type == 'task':
            self.__task_masks = torch.zeros(self.num_tasks + 1, self.num_classes).bool()
            for i in range(self.num_tasks):
                self.__task_masks[i, self.__splits[i]] = True
            
        if self.metric is not None:
            self.__evaluator = evaluator_map[self.metric](self.num_tasks, self.__task_ids)
        self.__test_results = []
        
    def _update_target_dataset(self):
        target_dataset = self.__graph.clone()
        if self.minimize:
            target_dataset = self.__graph.clone()
            for k, v in self.__cover_rule.items():
                if v == 'node': target_dataset.ndata.pop(k)
                elif v == 'edge': target_dataset.edata.pop(k)
                    
            target_dataset.ndata['feat'] = self.__graph.ndata['feat'].clone()
            target_dataset.ndata['label'] = self.__graph.ndata['label'].clone()
            target_dataset.ndata['train_mask'] = self.__graph.ndata['train_mask'].clone()
            target_dataset.ndata['val_mask'] = self.__graph.ndata['val_mask'].clone()
            target_dataset.ndata['test_mask'] = self.__graph.ndata['test_mask'].clone()
        target_dataset.ndata['train_mask'] = target_dataset.ndata['train_mask'] & (self.__task_ids == self._curr_task)
        target_dataset.ndata['val_mask'] = target_dataset.ndata['val_mask'] & (self.__task_ids == self._curr_task)
        if self.cover_unseen:
            target_dataset.ndata['label'][target_dataset.ndata['test_mask'] | (self.__task_ids > self._curr_task)] = -1
        
        if self.incr_type == 'class':
            self._target_dataset = target_dataset
        elif self.incr_type == 'task':
            self._target_dataset = target_dataset
            self._target_dataset.ndata['task_specific_mask'] = self.__task_masks[self.__task_ids]
        elif self.incr_type == 'time':
            srcs, dsts = target_dataset.edges()
            nodes_ready = self.__task_ids <= self._curr_task
            edges_ready = (self.__task_ids[srcs] <= self._curr_task) & (self.__task_ids[dsts] <= self._curr_task)
            self._target_dataset = dgl.graph((srcs[edges_ready], dsts[edges_ready]), num_nodes=self.__graph.num_nodes())
            for k in target_dataset.ndata.keys():
                self._target_dataset.ndata[k] = target_dataset.ndata[k]
                if self._target_dataset.ndata[k].dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
                    self._target_dataset.ndata[k][~nodes_ready] = -1
                else:
                    self._target_dataset.ndata[k][~nodes_ready] = 0

            for k in target_dataset.edata.keys():
                self._target_dataset.edata[k] = target_dataset.edata[k][edges_ready]
            self._target_dataset.ndata['test_mask'] = self._target_dataset.ndata['test_mask'] & (self.__task_ids <= self._curr_task)
            # self._target_dataset = dgl.add_self_loop(dgl.remove_self_loop(self._target_dataset))
        elif self.incr_type == 'domain':
            self._target_dataset = target_dataset
            
    def _update_accumulated_dataset(self):
        if self.minimize:
            target_dataset = self.__graph.clone()
            for k, v in self.__cover_rule.items():
                if v == 'node': target_dataset.ndata.pop(k)
                elif v == 'edge': target_dataset.edata.pop(k)
                    
            target_dataset.ndata['feat'] = self.__graph.ndata['feat'].clone()
            target_dataset.ndata['label'] = self.__graph.ndata['label'].clone()
            target_dataset.ndata['train_mask'] = self.__graph.ndata['train_mask'].clone()
            target_dataset.ndata['val_mask'] = self.__graph.ndata['val_mask'].clone()
            target_dataset.ndata['test_mask'] = self.__graph.ndata['test_mask'].clone()
        else:
            target_dataset = copy.deepcopy(self.__graph)
            
        target_dataset.ndata['train_mask'] = target_dataset.ndata['train_mask'] & (self.__task_ids <= self._curr_task)
        target_dataset.ndata['val_mask'] = target_dataset.ndata['val_mask'] & (self.__task_ids <= self._curr_task)
        if self.cover_unseen:
            target_dataset.ndata['label'][target_dataset.ndata['test_mask'] | (self.__task_ids > self._curr_task)] = -1
        
        if self.incr_type == 'class':
            self._accumulated_dataset = target_dataset
        elif self.incr_type == 'task':
            self._accumulated_dataset = target_dataset
            self._accumulated_dataset.ndata['task_specific_mask'] = self.__task_masks[self.__task_ids]
        elif self.incr_type == 'time':
            srcs, dsts = target_dataset.edges()
            nodes_ready = self.__task_ids <= self._curr_task
            edges_ready = (self.__task_ids[srcs] <= self._curr_task) & (self.__task_ids[dsts] <= self._curr_task)
            self._accumulated_dataset = dgl.graph((srcs[edges_ready], dsts[edges_ready]), num_nodes=self.__graph.num_nodes())
            for k in target_dataset.ndata.keys():
                self._accumulated_dataset.ndata[k] = target_dataset.ndata[k]
                if self._accumulated_dataset.ndata[k].dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
                    self._accumulated_dataset.ndata[k][~nodes_ready] = -1
                else:
                    self._accumulated_dataset.ndata[k][~nodes_ready] = 0

            for k in target_dataset.edata.keys():
                self._accumulated_dataset.edata[k] = target_dataset.edata[k][edges_ready]
            self._accumulated_dataset.ndata['test_mask'] = self._accumulated_dataset.ndata['test_mask'] & (self.__task_ids <= self._curr_task)
            # self._accumulated_dataset = dgl.add_self_loop(dgl.remove_self_loop(self._accumulated_dataset))
        elif self.incr_type == 'domain':
            self._accumulated_dataset = target_dataset
            
    def _get_eval_result_inner(self, preds, target_split):
        gt = self.__graph.ndata['label'][self._target_dataset.ndata[target_split + '_mask']]
        assert preds.shape == gt.shape, "shape mismatch"
        return self.__evaluator(preds, gt, torch.arange(self._target_dataset.num_nodes())[self._target_dataset.ndata[target_split + '_mask']])
    
    def get_eval_result(self, preds, target_split='test'):
        return self._get_eval_result_inner(preds, target_split)
    
    def get_accum_eval_result(self, preds, target_split='test'):
        gt = self.__graph.ndata['label'][self._accumulated_dataset.ndata[target_split + '_mask']]
        assert preds.shape == gt.shape, "shape mismatch"
        return self.__evaluator(preds, gt, torch.arange(self._accumulated_dataset.num_nodes())[self._accumulated_dataset.ndata[target_split + '_mask']])
        # return self._get_eval_result_inner(preds, target_split)
    
    def get_simple_eval_result(self, curr_batch_preds, curr_batch_gts):
        return self.__evaluator.simple_eval(curr_batch_preds, curr_batch_gts)
    
    def next_task(self, preds=torch.empty(1)):
        self.__test_results.append(self._get_eval_result_inner(preds, target_split='test'))
        super().next_task(preds)
        if self._curr_task == self.num_tasks: return self.__test_results

class DGLLinkPredictionIL(DGLBasicIL):
    def _init_continual_scenario(self):
        self.num_feats, self.__graph, self.__neg_edges = load_link_dataset(self.dataset_name, self.save_path)
        self.num_classes = 1
        
        if self.incr_type in ['class', 'task']:
            raise NotImplementedError
        elif self.incr_type == 'time':
            pkl_path = os.path.join(self.save_path, f'{self.dataset_name}_metadata_timeIL.pkl')
            download(f'https://github.com/anonymous-submission-23/anonymous-submission-23.github.io/raw/main/_splits/{self.dataset_name}_metadata_timeIL.pkl', pkl_path)
            metadata = pickle.load(open(pkl_path, 'rb'))
            self.__inner_tvt_splits = metadata['inner_tvt_splits']
            self.__time_splits = metadata['time_splits']
            
            self.num_tasks = len(self.__time_splits) - 1
            self.__task_ids = torch.zeros_like(self.__graph.edata['time'])
            for i in range(1, self.num_tasks):
                self.__task_ids[self.__graph.edata['time'] >= self.__time_splits[i]] = i
        elif self.incr_type == 'domain':
            pkl_path = os.path.join(self.save_path, f'{self.dataset_name}_metadata_domainIL.pkl')
            download(f'https://github.com/anonymous-submission-23/anonymous-submission-23.github.io/raw/main/_splits/{self.dataset_name}_metadata_domainIL.pkl', pkl_path)
            metadata = pickle.load(open(pkl_path, 'rb'))
            self.__inner_tvt_splits = metadata['inner_tvt_splits']
            self.__neg_edges = metadata['neg_edges']
            
            self.num_tasks = min(self.num_tasks, self.__graph.ndata['domain'].max().item() + 1)
            if self.kwargs is not None and 'task_shuffle' in self.kwargs and self.kwargs['task_shuffle']:
                domain_order = torch.randperm(self.num_tasks)
            else:
                domain_order = torch.arange(self.num_tasks)
            domain_order_inv = torch.arange(self.num_tasks)
            domain_order_inv[domain_order] = torch.arange(self.num_tasks)
            
            domain_infos = self.__graph.ndata.pop('domain')
            srcs, dsts = self.__graph.edges()
            self.__task_ids = torch.max(domain_order_inv[domain_infos[srcs]], domain_order_inv[domain_infos[dsts]])
            print(domain_order, self.num_tasks)
            print(torch.bincount(self.__task_ids), torch.cumsum(torch.bincount(self.__task_ids), dim=-1))
            print(torch.cumsum(torch.cumsum(torch.bincount(domain_order_inv[domain_infos[srcs]] * 10 + domain_order_inv[domain_infos[dsts]]).view(10, 10), dim=1), dim=0)[range(10),range(10)])
            
        """
        elif self.incr_type == 'domain':
            if self.kwargs is not None and 'task_shuffle' in self.kwargs and self.kwargs['task_shuffle']:
                pr_order = torch.randperm(self.num_feats)
            else:
                pr_order = torch.arange(self.num_feats)
            pr_order_inv = torch.arange(self.num_feats)
            pr_order_inv[pr_order] = torch.arange(self.num_feats)
            print('pr_order:', pr_order.numpy().tolist())
            
            pr_infos = self.__graph.ndata.pop('domain')
            srcs, dsts = self.__graph.edges()
            counts = torch.bincount(pr_infos[srcs] * self.num_feats + pr_infos[dsts], minlength=self.num_feats * self.num_feats).view(self.num_feats, self.num_feats)
            ordered_cumcounts = torch.cumsum(torch.cumsum(counts[pr_order][:, pr_order], dim=1), dim=0)[torch.arange(self.num_feats), torch.arange(self.num_feats)]
            print(ordered_cumcounts)
            node_task_ids = -torch.ones(self.num_feats, dtype=torch.long)
            for i in range(self.num_feats):
                node_task_ids[pr_order[i]] = max(ordered_cumcounts[i] * self.num_tasks - 1, 0) // ordered_cumcounts[-1]
            self.__task_ids = torch.max(node_task_ids[pr_infos[srcs]], node_task_ids[pr_infos[dsts]])
            self.__task_ids_neg = {_split: node_task_ids[pr_infos[self.__neg_edges[_split + '_pos']]].max(-1).values for _split in ['val', 'test']}
            # print((self.__task_ids[0::2] - self.__task_ids[1::2]).abs().max())
            # print(torch.cumsum(torch.bincount(torch.max(pr_order_inv[pr_infos[srcs]], pr_order_inv[pr_infos[dsts]])), dim=-1))
        """
        
        if self.metric is not None:
            if '@' in self.metric:
                metric_name, metric_k = self.metric.split('@')
                self.__evaluator = evaluator_map[metric_name](self.num_tasks, int(metric_k))
            else:
                self.__evaluator = evaluator_map[self.metric](self.num_tasks, self.__task_ids)
        self.__test_results = []
        
    def _update_target_dataset(self):
        srcs, dsts = self.__graph.edges()
        
        is_even = ((torch.arange(self.__inner_tvt_splits.shape[0]) % 2) == 0)
        edges_for_train = (self.__inner_tvt_splits < 8)
        if self.incr_type == 'time':
            edges_for_train &= (self.__task_ids <= self._curr_task)
        edges_ready = {'val': ((self.__inner_tvt_splits == 8) & (self.__task_ids == self._curr_task)) & is_even,
                       'test': (self.__inner_tvt_splits > 8) & is_even}
        
        """
        elif self.incr_type == 'domain':
            is_even = ((torch.arange(self.__task_ids.shape[0]) % 2) == 0)
            edges_for_train = torch.ones_like(self.__task_ids, dtype=torch.bool) 
            edges_ready = {'val': (self.__task_ids_neg['val'] == self._curr_task),
                           'test': torch.ones(self.__task_ids_neg['test'].shape[0], dtype=torch.bool)}
        """
        
        target_dataset = dgl.graph((srcs[edges_for_train], dsts[edges_for_train]), num_nodes=self.__graph.num_nodes())
        for k in self.__graph.ndata.keys():
            if (not self.minimize) or (k != 'time' or k != 'domain'): target_dataset.ndata[k] = self.__graph.ndata[k]
        for k in self.__graph.edata.keys():
            if (not self.minimize) or (k != 'time' or k != 'domain'): target_dataset.edata[k] = self.__graph.edata[k][edges_for_train]

        target_edges = {_split: torch.stack((srcs[edges_ready[_split]], dsts[edges_ready[_split]]), dim=-1) for _split in ['val', 'test']}
        gt_labels = {_split: torch.cat((self.__task_ids[edges_ready[_split]] + 1,
                             torch.zeros(self.__neg_edges[_split].shape[0], dtype=torch.long)), dim=0) for _split in ['val', 'test']}
        
        """
        elif self.incr_type == 'domain':
            target_edges = {_split: self.__neg_edges[_split + '_pos'][edges_ready[_split]] for _split in ['val', 'test']}
            gt_labels = {_split: torch.cat((self.__task_ids_neg[_split][edges_ready[_split]] + 1,
                                 torch.zeros(self.__neg_edges[_split].shape[0], dtype=torch.long)), dim=0) for _split in ['val', 'test']}
        """
        
        randperms = {_split: torch.randperm(gt_labels[_split].shape[0]) for _split in ['val', 'test']}
        target_edges = {_split: torch.cat((target_edges[_split], self.__neg_edges[_split]), dim=0)[randperms[_split]] for _split in ['val', 'test']}

        edges_ready['train'] = (edges_for_train & is_even) & (self.__task_ids == self._curr_task)
        target_edges['train'] = torch.stack((srcs[edges_ready['train']], dsts[edges_ready['train']]), dim=-1)

        self.__target_labels = {_split: gt_labels[_split][randperms[_split]] for _split in ['val', 'test']}
        self._target_dataset = {'graph': dgl.add_self_loop(target_dataset),
                                'train': {'edge': target_edges['train']},
                                'val': {'edge': target_edges['val'], 'label': (self.__target_labels['val'] > 0).long()},
                                'test': {'edge': target_edges['test'], 'label': -torch.ones_like(self.__target_labels['test'])}}
        self._target_dataset['train']['label'] = torch.ones(self._target_dataset['train']['edge'].shape[0], dtype=torch.long)
        
    def _update_accumulated_dataset(self):
        srcs, dsts = self.__graph.edges()
        
        is_even = ((torch.arange(self.__inner_tvt_splits.shape[0]) % 2) == 0)
        edges_for_train = (self.__inner_tvt_splits < 8)
        if self.incr_type == 'time':
            edges_for_train &= (self.__task_ids <= self._curr_task)
        edges_ready = {'val': ((self.__inner_tvt_splits == 8) & (self.__task_ids <= self._curr_task)) & is_even,
                       'test': (self.__inner_tvt_splits > 8) & is_even}
        """
        elif self.incr_type == 'domain':
            is_even = ((torch.arange(self.__task_ids.shape[0]) % 2) == 0)
            edges_for_train = torch.ones_like(self.__task_ids, dtype=torch.bool) 
            edges_ready = {'val': (self.__task_ids_neg['val'] <= self._curr_task),
                           'test': torch.ones(self.__task_ids_neg['test'].shape[0], dtype=torch.bool)}
        """
        
        target_dataset = dgl.graph((srcs[edges_for_train], dsts[edges_for_train]), num_nodes=self.__graph.num_nodes())
        for k in self.__graph.ndata.keys():
            if (not self.minimize) or (k != 'time' or k != 'domain'): target_dataset.ndata[k] = self.__graph.ndata[k]
        for k in self.__graph.edata.keys():
            if (not self.minimize) or (k != 'time' or k != 'domain'): target_dataset.edata[k] = self.__graph.edata[k][edges_for_train]
            
        target_edges = {_split: torch.stack((srcs[edges_ready[_split]], dsts[edges_ready[_split]]), dim=-1) for _split in ['val']}
        gt_labels = {_split: torch.cat((self.__task_ids[edges_ready[_split]] + 1,
                             torch.zeros(self.__neg_edges[_split].shape[0], dtype=torch.long)), dim=0) for _split in ['val']}
        """
        elif self.incr_type == 'domain':
            target_edges = {_split: self.__neg_edges[_split + '_pos'][edges_ready[_split]] for _split in ['val']}
            gt_labels = {_split: torch.cat((self.__task_ids_neg[_split][edges_ready[_split]] + 1,
                                 torch.zeros(self.__neg_edges[_split].shape[0], dtype=torch.long)), dim=0) for _split in ['val']}
        """
        
        randperms = {_split: torch.randperm(gt_labels[_split].shape[0]) for _split in ['val']}
        target_edges = {_split: torch.cat((target_edges[_split], self.__neg_edges[_split]), dim=0)[randperms[_split]] for _split in ['val']}

        edges_ready['train'] = (edges_for_train & is_even) & (self.__task_ids <= self._curr_task)
        target_edges['train'] = torch.stack((srcs[edges_ready['train']], dsts[edges_ready['train']]), dim=-1)

        self.__accumulated_labels = {_split: gt_labels[_split][randperms[_split]] for _split in ['val']}
        self.__accumulated_labels['test'] = self.__target_labels['test']
        self._accumulated_dataset = {'graph': dgl.add_self_loop(target_dataset),
                                     'train': {'edge': target_edges['train']},
                                     'val': {'edge': target_edges['val'], 'label': (self.__accumulated_labels['val'] > 0).long()},
                                     'test': self._target_dataset['test']}
        self._accumulated_dataset['train']['label'] = torch.ones(self._accumulated_dataset['train']['edge'].shape[0], dtype=torch.long)
        
    def _get_eval_result_inner(self, preds, target_split):
        gt = (self.__target_labels[target_split] > 0).long()
        assert preds.shape == gt.shape, "shape mismatch"
        return self.__evaluator(preds, gt, self.__target_labels[target_split] - 1)
    
    def get_eval_result(self, preds, target_split='test'):
        return self._get_eval_result_inner(preds, target_split)
    
    def get_accum_eval_result(self, preds, target_split='test'):
        gt = (self.__accumulated_labels[target_split] > 0).long()
        assert preds.shape == gt.shape, "shape mismatch"
        return self.__evaluator(preds, gt, self.__accumulated_labels[target_split] - 1)
        
    def get_simple_eval_result(self, curr_batch_preds, curr_batch_gts):
        return self.__evaluator.simple_eval(curr_batch_preds, curr_batch_gts)
    
    def next_task(self, preds=torch.empty(1)):
        self.__test_results.append(self._get_eval_result_inner(preds, target_split='test'))
        super().next_task(preds)
        if self._curr_task == self.num_tasks: return self.__test_results
    
class DGLGraphClassificationIL(DGLBasicIL):
    def _init_continual_scenario(self):
        self.num_classes, self.num_feats, self.__dataset = load_graph_dataset(self.dataset_name, self.save_path)
        
        if self.incr_type in ['class', 'task']:
            if self.kwargs is not None and 'task_orders' in self.kwargs:
                self.__splits = tuple([torch.LongTensor(class_ids) for class_ids in self.kwargs['task_orders']])
            elif self.kwargs is not None and 'task_shuffle' in self.kwargs and self.kwargs['task_shuffle']:
                self.__splits = torch.split(torch.randperm(self.num_classes), self.num_classes // self.num_tasks)[:self.num_tasks]
            else:
                self.__splits = torch.split(torch.arange(self.num_classes), self.num_classes // self.num_tasks)[:self.num_tasks]
            print('class split information:', self.__splits)
            id_to_task = self.num_tasks * torch.ones(self.num_classes).long()
            for i in range(self.num_tasks):
                id_to_task[self.__splits[i]] = i
            self.__task_ids = id_to_task[self.__dataset._labels]
            self.__original_labels = self.__dataset._labels.clone()
            self.__dataset._labels[self.__dataset._test_masks] = -1
        elif self.incr_type == 'time':
            pkl_path = os.path.join(self.save_path, f'{self.dataset_name}_metadata_timeIL.pkl')
            download(f'https://github.com/anonymous-submission-23/anonymous-submission-23.github.io/raw/main/_splits/{self.dataset_name}_metadata_timeIL.pkl', pkl_path)
            metadata = pickle.load(open(pkl_path, 'rb'))
            inner_tvt_splits = metadata['inner_tvt_splits']
            self.__time_splits = metadata['time_splits']
            # self.__time_splits = (self.__dataset._months - 1)
            # inner_tvt_splits = torch.randperm(self.__time_splits.shape[0])
            self.__dataset._train_masks = (inner_tvt_splits % 10) < 6
            self.__dataset._val_masks = ((inner_tvt_splits % 10) == 6) | ((inner_tvt_splits % 10) == 7) 
            self.__dataset._test_masks = (inner_tvt_splits % 10) > 7
            print(self.__dataset._train_masks.sum(), self.__dataset._val_masks.sum(), self.__dataset._test_masks.sum())
            
            self.num_tasks = self.__time_splits.max().item() + 1
            self.__task_ids = self.__time_splits
            self.__dataset._labels = self.__dataset._labels.squeeze()
            self.__original_labels = self.__dataset._labels.clone()
            self.__dataset._labels[self.__dataset._test_masks] = -1
            print(torch.bincount(self.__original_labels.squeeze()))
            
        elif self.incr_type == 'domain':
            pkl_path = os.path.join(self.save_path, f'{self.dataset_name}_metadata_domainIL.pkl')
            download(f'https://github.com/anonymous-submission-23/anonymous-submission-23.github.io/raw/main/_splits/{self.dataset_name}_metadata_domainIL.pkl', pkl_path)
            metadata = pickle.load(open(pkl_path, 'rb'))
            
            inner_tvt_splits = metadata['inner_tvt_splits']
            domain_info = metadata['domain_splits']
            self.num_tasks = domain_info.max().item() + 1
            
            if self.kwargs is not None and 'task_shuffle' in self.kwargs and self.kwargs['task_shuffle']:
                self.__task_order = torch.randperm(self.num_tasks)
                print('domain_order:', self.__task_order)
                self.__task_ids = self.__task_order[domain_info]
            else:
                self.__task_ids = domain_info
            
            self.__dataset._train_masks = (inner_tvt_splits % 10) < 8
            self.__dataset._val_masks = (inner_tvt_splits % 10) == 8
            self.__dataset._test_masks = (inner_tvt_splits % 10) > 8
            self.__original_labels = self.__dataset.labels.clone()
            self.__dataset.labels[self.__dataset._test_masks] = -1
            
        if self.incr_type == 'task':
            self.__task_masks = torch.zeros(self.num_tasks + 1, self.num_classes).bool()
            for i in range(self.num_tasks):
                self.__task_masks[i, self.__splits[i]] = True
            self.__dataset._task_specific_masks = self.__task_masks[self.__task_ids]
            
        if self.metric is not None:
            self.__evaluator = evaluator_map[self.metric](self.num_tasks, self.__task_ids)
        self.__test_results = []
        
    def _update_target_dataset(self):
        target_train_indices = torch.nonzero((self.__task_ids == self._curr_task) & self.__dataset._train_masks, as_tuple=True)[0]
        target_val_indices = torch.nonzero((self.__task_ids == self._curr_task) & self.__dataset._val_masks, as_tuple=True)[0]
        target_test_indices = torch.nonzero(self.__dataset._test_masks, as_tuple=True)[0]
        self._target_dataset = {'train': Subset(self.__dataset, target_train_indices), 'val': Subset(self.__dataset, target_val_indices), 'test': Subset(self.__dataset, target_test_indices)}
        
    def _update_accumulated_dataset(self):
        target_train_indices = torch.nonzero((self.__task_ids <= self._curr_task) & self.__dataset._train_masks, as_tuple=True)[0]
        target_val_indices = torch.nonzero((self.__task_ids <= self._curr_task) & self.__dataset._val_masks, as_tuple=True)[0]
        target_test_indices = torch.nonzero(self.__dataset._test_masks, as_tuple=True)[0]
        self._accumulated_dataset = {'train': Subset(self.__dataset, target_train_indices), 'val': Subset(self.__dataset, target_val_indices), 'test': Subset(self.__dataset, target_test_indices)}
        
    def _get_eval_result_inner(self, preds, target_split):
        gt = self.__original_labels[self._target_dataset[target_split].indices]
        assert preds.shape == gt.shape, "shape mismatch"
        return self.__evaluator(preds, gt, self._target_dataset[target_split].indices)
    
    def get_eval_result(self, preds, target_split='test'):
        return self._get_eval_result_inner(preds, target_split)
    
    def get_accum_eval_result(self, preds, target_split='test'):
        gt = self.__original_labels[self._accumulated_dataset[target_split].indices]
        assert preds.shape == gt.shape, "shape mismatch"
        return self.__evaluator(preds, gt, self._accumulated_dataset[target_split].indices)
    
    def get_simple_eval_result(self, curr_batch_preds, curr_batch_gts):
        return self.__evaluator.simple_eval(curr_batch_preds, curr_batch_gts)
    
    def next_task(self, preds=torch.empty(1)):
        self.__test_results.append(self._get_eval_result_inner(preds, target_split='test'))
        super().next_task(preds)
        if self._curr_task == self.num_tasks: return self.__test_results