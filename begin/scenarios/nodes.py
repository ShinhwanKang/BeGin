import torch
import dgl
import os
import pickle
import copy
from dgl.data.utils import download, Subset
from ogb.nodeproppred import DglNodePropPredDataset
from torch_scatter import scatter

from .common import BaseScenarioLoader
from .datasets import *
from . import evaluator_map

def load_node_dataset(dataset_name, dataset_load_func, incr_type, save_path):
    """
        The function for load node-level datasets.
    """
    cover_rule = {'feat': 'node', 'label': 'node', 'train_mask': 'node', 'val_mask': 'node', 'test_mask': 'node'}
    if dataset_load_func is not None:
        custom_dataset = dataset_load_func(save_path=save_path)
        graph = custom_dataset['graph']
        num_feats = custom_dataset['num_feats']
        num_classes = custom_dataset['num_classes']
    elif dataset_name in ['cora'] and incr_type in ['task', 'class']:
        dataset = dgl.data.CoraGraphDataset(raw_dir=save_path, verbose=False)
        graph = dataset._g
        num_feats, num_classes = graph.ndata['feat'].shape[-1], dataset.num_classes
    elif dataset_name in ['citeseer'] and incr_type in ['task', 'class']:
        dataset = dgl.data.CiteseerGraphDataset(raw_dir=save_path)
        graph = dataset._g
        num_feats, num_classes = graph.ndata['feat'].shape[-1], dataset.num_classes
    elif dataset_name in ['corafull'] and incr_type in ['task', 'class']:
        dataset = dgl.data.CoraFullDataset(raw_dir=save_path, verbose=False)
        graph = dataset._graph
        num_feats, num_classes = graph.ndata['feat'].shape[-1], dataset.num_classes
        
        # We need to designate train/val/test split since DGL does not provide the information.
        # We used random train/val/test split (6 : 2 : 2)
        pkl_path = os.path.join(save_path, f'corafull_metadata_allIL.pkl')
        download(f'https://github.com/anonymous-submission-23/anonymous-submission-23.github.io/raw/main/_splits/corafull_metadata_allIL.pkl', pkl_path)
        metadata = pickle.load(open(pkl_path, 'rb'))
        inner_tvt_splits = metadata['inner_tvt_splits'] % 10
        graph.ndata['train_mask'] = (inner_tvt_splits < 6)
        graph.ndata['val_mask'] = (6 <= inner_tvt_splits) & (inner_tvt_splits < 8)
        graph.ndata['test_mask'] = (8 <= inner_tvt_splits)
        
    elif dataset_name in ['ogbn-arxiv'] and incr_type in ['task', 'class', 'time']:
        dataset = DglNodePropPredDataset('ogbn-arxiv', root=save_path)
        graph, label = dataset[0]
        num_feats, num_classes = graph.ndata['feat'].shape[-1], dataset.num_classes
        
        # to_bidirected
        srcs, dsts = graph.all_edges()
        graph.add_edges(dsts, srcs)
        
        if incr_type == 'time':
            # load train/val/test split
            pkl_path = os.path.join(save_path, f'ogbn-arxiv_metadata_timeIL.pkl')
            download(f'https://github.com/anonymous-submission-23/anonymous-submission-23.github.io/raw/main/_splits/ogbn-arxiv_metadata_timeIL.pkl', pkl_path)
            metadata = pickle.load(open(pkl_path, 'rb'))
            inner_tvt_splits = metadata['inner_tvt_splits']
            graph.ndata['train_mask'] = (inner_tvt_splits < 4)
            graph.ndata['val_mask'] = (inner_tvt_splits == 4)
            graph.ndata['test_mask'] = (inner_tvt_splits > 4)
            # time information for splitting tasks
            graph.ndata['time'] = torch.clamp(graph.ndata.pop('year').squeeze() - 1997, 0, 20000)
        else:
            # load train/val/test split
            split_idx = dataset.get_idx_split()
            for _split, _split_name in [('train', 'train'), ('valid', 'val'), ('test', 'test')]:
                _indices = torch.zeros(graph.num_nodes(), dtype=torch.bool)
                _indices[split_idx[_split]] = True
                graph.ndata[_split_name + '_mask'] = _indices
        
        # load target label and timestamp information
        graph.ndata['label'] = label.squeeze()
        
    elif dataset_name in ['ogbn-products'] and incr_type in ['task', 'class']:
        dataset = DglNodePropPredDataset('ogbn-products', root=save_path)
        graph, label = dataset[0]
        num_feats, num_classes = graph.ndata['feat'].shape[-1], dataset.num_classes
        
        # load train/val/test split
        split_idx = dataset.get_idx_split()
        for _split, _split_name in [('train', 'train'), ('valid', 'val'), ('test', 'test')]:
            _indices = torch.zeros(graph.num_nodes(), dtype=torch.bool)
            _indices[split_idx[_split]] = True
            graph.ndata[_split_name + '_mask'] = _indices
        
        # load target label and timestamp information
        graph.ndata['label'] = label.squeeze()
    elif dataset_name in ['ogbn-proteins'] and incr_type in ['domain']:
        dataset = DglNodePropPredDataset('ogbn-proteins', root=save_path)
        graph, label = dataset[0]
        
        # create node features using edge features + load species information
        # (See https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/proteins/gnn.py : commit d04eada)
        graph.ndata['feat'] = scatter(graph.edata.pop('feat'), graph.edges()[0], dim=0, reduce='mean')
        unique_ids = torch.unique(graph.ndata['species'])
        raw_species_to_domain = -torch.ones(unique_ids.max().item() + 1, dtype=torch.long)
        raw_species_to_domain[unique_ids] = torch.arange(8)
        graph.ndata['species'] = raw_species_to_domain[graph.ndata.pop('species').squeeze(-1)]
        num_feats, num_classes = graph.ndata['feat'].shape[-1], label.shape[-1]
        
        # load train/val/test split
        pkl_path = os.path.join(save_path, f'ogbn-proteins_metadata_domainIL.pkl')
        download(f'https://github.com/anonymous-submission-23/anonymous-submission-23.github.io/raw/main/_splits/ogbn-proteins_metadata_domainIL.pkl', pkl_path)
        metadata = pickle.load(open(pkl_path, 'rb'))
        inner_tvt_splits = metadata['inner_tvt_splits']
        graph.ndata['train_mask'] = (inner_tvt_splits < 4)
        graph.ndata['val_mask'] = (inner_tvt_splits == 4)
        graph.ndata['test_mask'] = (inner_tvt_splits > 4)
        
        # load target label and domain information
        graph.ndata['label'] = label
        if incr_type == 'domain': graph.ndata['domain'] = graph.ndata.pop('species').squeeze()
        
    elif dataset_name in ['ogbn-mag'] and incr_type in ['task', 'class', 'time']:
        dataset = DglNodePropPredDataset('ogbn-mag', root=save_path)
        _graph, _label = dataset[0]
        srcs, dsts = _graph.edges(etype='cites')
        graph = dgl.graph((srcs, dsts))
        
        # pick nodes whose entity is 'paper'
        graph.ndata['feat'] = _graph.ndata['feat']['paper']
        graph.add_edges(dsts, srcs)
        label = _label['paper'].squeeze()
        
        split_idx = dataset.get_idx_split()
        
        # (for task, class) select classes with at least 10 nodes (in train, valid, and test split)
        if incr_type in ['task', 'class']:
            traincnt = torch.bincount(label[split_idx['train']['paper']])
            valcnt = torch.bincount(label[split_idx['valid']['paper']])
            testcnt = torch.bincount(label[split_idx['test']['paper']])
            considered_labels = torch.nonzero(torch.min(torch.stack((traincnt, valcnt, testcnt), dim=-1), dim=-1).values >= 10, as_tuple=True)[0]
            processed_labels = torch.ones(label.max() + 1, dtype=torch.long) * considered_labels.shape[0]
            processed_labels[considered_labels] = torch.arange(considered_labels.shape[0])
            label = processed_labels[label]
            num_feats, num_classes = graph.ndata['feat'].shape[-1], label.max().item() # ignore the last class
            
            # load train/val/test split
            for _split, _split_name in [('train', 'train'), ('valid', 'val'), ('test', 'test')]:
                _indices = torch.zeros(graph.num_nodes(), dtype=torch.bool)
                _indices[split_idx[_split]['paper']] = True
                graph.ndata[_split_name + '_mask'] = _indices
            graph.ndata['label'] = label.squeeze()
        elif incr_type in ['time']:
            pkl_path = os.path.join(save_path, f'ogbn-mag_metadata_timeIL.pkl')
            download(f'https://github.com/anonymous-submission-23/anonymous-submission-23.github.io/raw/main/_splits/ogbn-mag_metadata_timeIL.pkl', pkl_path)
            metadata = pickle.load(open(pkl_path, 'rb'))
            inner_tvt_splits = metadata['inner_tvt_splits']
            graph.ndata['train_mask'] = (inner_tvt_splits < 4)
            graph.ndata['val_mask'] = (inner_tvt_splits == 4)
            graph.ndata['test_mask'] = (inner_tvt_splits > 4)
            
            graph.ndata['label'] = label.squeeze()
            num_feats, num_classes = graph.ndata['feat'].shape[-1], (label.max().item() + 1)
            graph.ndata['time'] = _graph.ndata['year']['paper'].squeeze() - 2010
            
    elif dataset_name in ['twitch'] and incr_type in ['domain']:
        dataset = TwitchGamerNodeDataset('twitch', raw_dir=save_path)
        graph = dataset[0]
        num_feats, num_classes = graph.ndata['feat'].shape[-1], dataset.num_classes
        
        pkl_path = os.path.join(save_path, f'twitch_metadata_domainIL.pkl')
        download(f'https://github.com/jihoon-ko/BeGin/raw/main/metadata/twitch_metadata_domainIL.pkl', pkl_path)
        metadata = pickle.load(open(pkl_path, 'rb'))
        inner_tvt_splits = metadata['inner_tvt_splits']
        graph.ndata['train_mask'] = (inner_tvt_splits < 4)
        graph.ndata['val_mask'] = (inner_tvt_splits == 4)
        graph.ndata['test_mask'] = (inner_tvt_splits > 4)
            
    else:
        raise NotImplementedError("Tried to load unsupported scenario.")
        
    # We hide information of unseen nodes (for Time-IL) 
    for k in graph.ndata.keys():
        if k not in cover_rule:
            cover_rule[k] = 'node'
    for k in graph.edata.keys():
        if k not in cover_rule:
            cover_rule[k] = 'edge'
    
    # remove and add self-loop (to prevent duplicates)
    srcs, dsts = graph.edges()
    is_non_loop = (srcs != dsts)
    final_graph = dgl.graph((srcs[is_non_loop], dsts[is_non_loop]), num_nodes=graph.num_nodes())
    for k in graph.ndata.keys():
        final_graph.ndata[k] = graph.ndata[k]
    for k in graph.edata.keys():
        final_graph.edata[k] = graph.edata[k][is_non_loop]
    final_graph = dgl.add_self_loop(final_graph)
    
    print("=====CHECK=====")
    print("num_classes:", num_classes, ", num_feats:", num_feats)
    print("graph.ndata['train_mask']:", 'train_mask' in graph.ndata)
    print("graph.ndata['val_mask']:", 'val_mask' in graph.ndata)
    print("graph.ndata['test_mask']:", 'test_mask' in graph.ndata)
    print("graph.ndata['label']:", 'label' in graph.ndata)
    if incr_type == 'time':
        print("graph.ndata['time']:", 'time' in graph.ndata)
    if incr_type == 'domain':
        print("graph.ndata['domain']:", 'domain' in graph.ndata)
    print("===============")
    
    return num_classes, num_feats, final_graph, cover_rule

class NCScenarioLoader(BaseScenarioLoader):
    """
        The sceanario loader for node classification problems.

        **Usage example:**

            >>> scenario = NCScenarioLoader(dataset_name dataset_object=None, num_tasks=3, metric="accuracy", 
            ...                             save_path="./data", incr_type="task", task_shuffle=True)

        Bases: ``BaseScenarioLoader``
    """
    
    def _init_continual_scenario(self):
        self.num_classes, self.num_feats, self.__graph, self.__cover_rule = load_node_dataset(self.dataset_name, self.dataset_load_func, self.incr_type, self.save_path)
        if self.incr_type in ['class', 'task']:
            # determine task configuration
            if self.kwargs is not None and 'task_orders' in self.kwargs:
                self.__splits = tuple([torch.LongTensor(class_ids) for class_ids in self.kwargs['task_orders']])
            elif self.kwargs is not None and 'task_shuffle' in self.kwargs and self.kwargs['task_shuffle']:
                self.__splits = torch.split(torch.randperm(self.num_classes), self.num_classes // self.num_tasks)[:self.num_tasks]
            else:
                self.__splits = torch.split(torch.arange(self.num_classes), self.num_classes // self.num_tasks)[:self.num_tasks]
            
            print('class split information:', self.__splits)
            # compute task ids for each node
            id_to_task = self.num_tasks * torch.ones(self.__graph.ndata['label'].max() + 1).long()
            for i in range(self.num_tasks):
                id_to_task[self.__splits[i]] = i
            self.__task_ids = id_to_task[self.__graph.ndata['label']]
            
            # ignore classes which are not used in the tasks
            self.__graph.ndata['test_mask'] = self.__graph.ndata['test_mask'] & (self.__task_ids < self.num_tasks)
        elif self.incr_type == 'time':
            # compute task ids for each node
            self.__task_ids = self.__graph.ndata['time']
            if self.num_tasks != self.__task_ids.max().item() + 1:
                print("WARNING: Mismatch between the number of tasks and the processed data. Please check again.")
            # overwrite num_tasks
            self.num_tasks = self.__task_ids.max().item() + 1
        elif self.incr_type == 'domain':
            # num_tasks only depends on the number of domains
            self.num_tasks = self.__graph.ndata['domain'].max().item() + 1
            # determine task configuration
            if self.kwargs is not None and 'task_shuffle' in self.kwargs and self.kwargs['task_shuffle']:
                self.__task_order = torch.randperm(self.num_tasks)
                print('domain_order:', self.__task_order)
                self.__task_ids = self.__task_order[self.__graph.ndata['domain']]
            else:
                self.__task_ids = self.__graph.ndata['domain']
                
        # we need to provide task information (only for task-IL)
        if self.incr_type == 'task':
            self.__task_masks = torch.zeros(self.num_tasks + 1, self.num_classes).bool()
            for i in range(self.num_tasks):
                self.__task_masks[i, self.__splits[i]] = True
        
        # set evaluator for the target scenario
        if self.metric is not None:
            self.__evaluator = evaluator_map[self.metric](self.num_tasks, self.__task_ids)
        self.__test_results = []
        
    def _update_target_dataset(self):
        target_dataset = self.__graph.clone()
        
        # conceal unnecessary information
        for k, v in self.__cover_rule.items():
            if v == 'node': target_dataset.ndata.pop(k)
            elif v == 'edge': target_dataset.edata.pop(k)
        target_dataset.ndata['feat'] = self.__graph.ndata['feat'].clone()
        target_dataset.ndata['label'] = self.__graph.ndata['label'].clone()
        target_dataset.ndata['train_mask'] = self.__graph.ndata['train_mask'].clone()
        target_dataset.ndata['val_mask'] = self.__graph.ndata['val_mask'].clone()
        target_dataset.ndata['test_mask'] = self.__graph.ndata['test_mask'].clone()
        
        # update train/val/test mask for the current task
        target_dataset.ndata['train_mask'] = target_dataset.ndata['train_mask'] & (self.__task_ids == self._curr_task)
        target_dataset.ndata['val_mask'] = target_dataset.ndata['val_mask'] & (self.__task_ids == self._curr_task)
        target_dataset.ndata['label'][target_dataset.ndata['test_mask'] | (self.__task_ids > self._curr_task)] = -1
        
        if self.incr_type == 'class':
            # for class-IL, no need to change
            self._target_dataset = target_dataset
        elif self.incr_type == 'task':
            # for task-IL, we need task information. BeGin provide the information with 'task_specific_mask'
            self._target_dataset = target_dataset
            self._target_dataset.ndata['task_specific_mask'] = self.__task_masks[self.__task_ids]
        elif self.incr_type == 'time':
            # for time-IL, we need to hide unseen nodes and information at the current timestamp
            
            # remain only seen nodes and edges
            srcs, dsts = target_dataset.edges()
            nodes_ready = self.__task_ids <= self._curr_task
            edges_ready = (self.__task_ids[srcs] <= self._curr_task) & (self.__task_ids[dsts] <= self._curr_task)
            self._target_dataset = dgl.graph((srcs[edges_ready], dsts[edges_ready]), num_nodes=self.__graph.num_nodes())
            
            # cover the information of the unseen nodes/edges
            for k in target_dataset.ndata.keys():
                self._target_dataset.ndata[k] = target_dataset.ndata[k]
                if self._target_dataset.ndata[k].dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
                    self._target_dataset.ndata[k][~nodes_ready] = -1
                else:
                    self._target_dataset.ndata[k][~nodes_ready] = 0
            for k in target_dataset.edata.keys():
                self._target_dataset.edata[k] = target_dataset.edata[k][edges_ready]
            
            # update test mask (exclude unseen test nodes)
            self._target_dataset.ndata['test_mask'] = self._target_dataset.ndata['test_mask'] & (self.__task_ids <= self._curr_task)
        elif self.incr_type == 'domain':
            # for domain-IL, no need to change
            self._target_dataset = target_dataset
            
    def _update_accumulated_dataset(self):
        target_dataset = self.__graph.clone()
        
        # conceal unnecessary information
        for k, v in self.__cover_rule.items():
            if v == 'node': target_dataset.ndata.pop(k)
            elif v == 'edge': target_dataset.edata.pop(k)

        target_dataset.ndata['feat'] = self.__graph.ndata['feat'].clone()
        target_dataset.ndata['label'] = self.__graph.ndata['label'].clone()
        target_dataset.ndata['train_mask'] = self.__graph.ndata['train_mask'].clone()
        target_dataset.ndata['val_mask'] = self.__graph.ndata['val_mask'].clone()
        target_dataset.ndata['test_mask'] = self.__graph.ndata['test_mask'].clone()
        
        # update train/val/test mask for the current task
        target_dataset.ndata['train_mask'] = target_dataset.ndata['train_mask'] & (self.__task_ids <= self._curr_task)
        target_dataset.ndata['val_mask'] = target_dataset.ndata['val_mask'] & (self.__task_ids <= self._curr_task)
        target_dataset.ndata['label'][target_dataset.ndata['test_mask'] | (self.__task_ids > self._curr_task)] = -1
        
        if self.incr_type == 'class':
            # for class-IL, no need to change
            self._accumulated_dataset = target_dataset
        elif self.incr_type == 'task':
            # for task-IL, we need task information. BeGin provide the information with 'task_specific_mask'
            self._accumulated_dataset = target_dataset
            self._accumulated_dataset.ndata['task_specific_mask'] = self.__task_masks[self.__task_ids]
        elif self.incr_type == 'time':
            # for time-IL, we need to hide unseen nodes and information at the current timestamp
            srcs, dsts = target_dataset.edges()
            nodes_ready = self.__task_ids <= self._curr_task
            edges_ready = (self.__task_ids[srcs] <= self._curr_task) & (self.__task_ids[dsts] <= self._curr_task)
            self._accumulated_dataset = dgl.graph((srcs[edges_ready], dsts[edges_ready]), num_nodes=self.__graph.num_nodes())
            
            # cover the information of the unseen nodes/edges
            for k in target_dataset.ndata.keys():
                self._accumulated_dataset.ndata[k] = target_dataset.ndata[k]
                if self._accumulated_dataset.ndata[k].dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
                    self._accumulated_dataset.ndata[k][~nodes_ready] = -1
                else:
                    self._accumulated_dataset.ndata[k][~nodes_ready] = 0
            for k in target_dataset.edata.keys():
                self._accumulated_dataset.edata[k] = target_dataset.edata[k][edges_ready]
                
            # update test mask (exclude unseen test nodes)
            self._accumulated_dataset.ndata['test_mask'] = self._accumulated_dataset.ndata['test_mask'] & (self.__task_ids <= self._curr_task)
        elif self.incr_type == 'domain':
            self._accumulated_dataset = target_dataset
            
    def _get_eval_result_inner(self, preds, target_split):
        """
            The inner function of get_eval_result.
            
            Args:
                preds (torch.Tensor): predicted output of the current model
                target_split (str): target split to measure the performance (spec., 'val' or 'test')
        """
        gt = self.__graph.ndata['label'][self._target_dataset.ndata[target_split + '_mask']]
        assert preds.shape == gt.shape, "shape mismatch"
        return self.__evaluator(preds, gt, torch.arange(self._target_dataset.num_nodes())[self._target_dataset.ndata[target_split + '_mask']])
    
    def get_eval_result(self, preds, target_split='test'):
        return self._get_eval_result_inner(preds, target_split)
    
    def get_accum_eval_result(self, preds, target_split='test'):
        """ 
            Compute performance on the accumulated dataset for the given target split.
            It can be used to compute train/val performance during training.
            
            Args:
                preds (torch.Tensor): predicted output of the current model
                target_split (str): target split to measure the performance (spec., 'val' or 'test')
        """
        
        gt = self.__graph.ndata['label'][self._accumulated_dataset.ndata[target_split + '_mask']]
        assert preds.shape == gt.shape, "shape mismatch"
        return self.__evaluator(preds, gt, torch.arange(self._accumulated_dataset.num_nodes())[self._accumulated_dataset.ndata[target_split + '_mask']])
    
    def get_simple_eval_result(self, curr_batch_preds, curr_batch_gts):
        """ 
            Compute performance for the given batch when we ignore task configuration.
            It can be used to compute train/val performance during training.
            
            Args:
                curr_batch_preds (torch.Tensor): predicted output of the current model
                curr_batch_gts (torch.Tensor): ground-truth labels
        """
        return self.__evaluator.simple_eval(curr_batch_preds, curr_batch_gts)
    
    def next_task(self, preds=torch.empty(1)):
        if self.export_mode:
            super().next_task(preds)
        else:
            self.__test_results.append(self._get_eval_result_inner(preds, target_split='test'))
            super().next_task(preds)
            if self._curr_task == self.num_tasks:
                scores = torch.stack(self.__test_results, dim=0)
                scores_np = scores.detach().cpu().numpy()
                ap = scores_np[-1, :-1].mean().item()
                af = (scores_np[np.arange(self.num_tasks), np.arange(self.num_tasks)] - scores_np[-1, :-1]).sum().item() / (self.num_tasks - 1)
                if self.initial_test_result is not None:
                    fwt = (scores_np[np.arange(self.num_tasks-1), np.arange(self.num_tasks-1)+1] - self.initial_test_result.detach().cpu().numpy()[1:-1]).sum() / (self.num_tasks - 1)
                else:
                    fwt = None
                return {'exp_results': scores, 'AP': ap, 'AF': af, 'FWT': fwt}
    
    def get_current_dataset_for_export(self, _global=False):
        """
            Returns:
                The graph dataset the implemented model uses in the current task
        """
        target_graph = self.__graph if _global else self._target_dataset
        metadata = {'num_classes': self.num_classes, 'ndata_feat': self.__graph.ndata['feat'], 'task': self.__task_ids} if _global else {}
        if _global and self.incr_type == 'task':  metadata['task_specific_mask'] = self.__task_masks[self.__task_ids]
        metadata['edges'] = target_graph.edges()
        metadata['train_mask'] = target_graph.ndata['train_mask']
        metadata['val_mask'] = target_graph.ndata['val_mask']
        if _global: metadata['test_mask'] = target_graph.ndata['test_mask']
        metadata['label'] = copy.deepcopy(target_graph.ndata['label'])
        return metadata