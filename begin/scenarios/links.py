import torch
import dgl
import os
import pickle
import copy
from dgl.data.utils import download, Subset
from ogb.linkproppred import DglLinkPropPredDataset

from .common import BaseScenarioLoader
from .datasets import *
from . import evaluator_map

def load_linkp_dataset(dataset_name, incr_type, save_path):
    if dataset_name == 'ogbl-collab' and incr_type in ['time']:
        dataset = DglLinkPropPredDataset(dataset_name, root=save_path)
        # load edges and negative edges
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
        
        # generate graphs with all edges (including val/test)
        graph = dgl.graph((combined['edge'][:, 0], combined['edge'][:, 1]), num_nodes=train_graph.num_nodes())
        for k in combined.keys():
            if k != 'edge':
                if k == 'year': graph.edata['time'] = combined[k]
                else: graph.edata[k] = combined[k]
        for k in train_graph.ndata.keys():
            graph.ndata[k] = train_graph.ndata[k]
        _srcs, _dsts = map(lambda x: x.numpy().tolist(), graph.edges())
        edgeset = {(s, d) for s, d in zip(_srcs, _dsts)}
        
        # choose negative edges
        neg_edges = {}
        neg_edges['val'] = torch.LongTensor([[_s, _d] for _s, _d in zip(*zip(*split_edge['valid']['edge_neg'].numpy().tolist())) if (_s, _d) not in edgeset])
        neg_edges['test'] = torch.LongTensor([[_s, _d] for _s, _d in zip(*zip(*split_edge['test']['edge_neg'].numpy().tolist())) if (_s, _d) not in edgeset])
    elif dataset_name == 'wikics' and incr_type in ['domain']:
        dataset = WikiCSLinkDataset(raw_dir=save_path)
        graph = dataset._g
        # load negative edges later
        neg_edges = {}
    else:
        raise NotImplementedError("Tried to load unsupported scenario.")
    
    return graph.ndata['feat'].shape[-1], graph, neg_edges

def load_linkc_dataset(dataset_name, incr_type, save_path):
    if dataset_name == 'bitcoin' and incr_type in ['task', 'class', 'time']:
        dataset = BitcoinOTCDataset(dataset_name, raw_dir=save_path)
        graph = dataset[0]
    else:
        raise NotImplementedError("Tried to load unsupported scenario.")
        
    return graph.ndata['feat'].shape[-1], graph

class LPScenarioLoader(BaseScenarioLoader):
    """
        The sceanario loader for link prediction.

        **Usage example:**

            >>> scenario = LPScenarioLoader(dataset_name="ogbl-collab", num_tasks=3, metric="hits@50", 
            ...             save_path="/data", incr_type="time", task_shuffle=True)

        Bases: ``BaseScenarioLoader``
    """
    def _init_continual_scenario(self):
        self.num_feats, self.__graph, self.__neg_edges = load_linkp_dataset(self.dataset_name, self.incr_type, self.save_path)
        self.num_classes = 1
        
        if self.incr_type in ['class', 'task']:
            # It is impossible to make class-IL and task-IL setting
            raise NotImplementedError
        elif self.incr_type == 'time':
            # load time split and train/val/test split information
            pkl_path = os.path.join(self.save_path, f'{self.dataset_name}_metadata_timeIL.pkl')
            download(f'https://github.com/anonymous-submission-23/anonymous-submission-23.github.io/raw/main/_splits/{self.dataset_name}_metadata_timeIL.pkl', pkl_path)
            metadata = pickle.load(open(pkl_path, 'rb'))
            self.__inner_tvt_splits = metadata['inner_tvt_splits']
            self.__time_splits = metadata['time_splits']
            
            # compute task ids for each instance
            self.num_tasks = len(self.__time_splits) - 1
            self.__task_ids = torch.zeros_like(self.__graph.edata['time'])
            for i in range(1, self.num_tasks):
                self.__task_ids[self.__graph.edata['time'] >= self.__time_splits[i]] = i
        elif self.incr_type == 'domain':
            # load domain information and train/val/test split
            pkl_path = os.path.join(self.save_path, f'{self.dataset_name}_metadata_domainIL.pkl')
            download(f'https://github.com/anonymous-submission-23/anonymous-submission-23.github.io/raw/main/_splits/{self.dataset_name}_metadata_domainIL.pkl', pkl_path)
            metadata = pickle.load(open(pkl_path, 'rb'))
            self.__inner_tvt_splits = metadata['inner_tvt_splits']
            self.__neg_edges = metadata['neg_edges']
            
            # determine task configuration
            self.num_tasks = min(self.num_tasks, self.__graph.ndata['domain'].max().item() + 1)
            if self.kwargs is not None and 'task_shuffle' in self.kwargs and self.kwargs['task_shuffle']:
                domain_order = torch.randperm(self.num_tasks)
            else:
                domain_order = torch.arange(self.num_tasks)
            domain_order_inv = torch.arange(self.num_tasks)
            domain_order_inv[domain_order] = torch.arange(self.num_tasks)
            
            # compute task ids for each link (instance)
            domain_infos = self.__graph.ndata.pop('domain')
            srcs, dsts = self.__graph.edges()
            self.__task_ids = torch.max(domain_order_inv[domain_infos[srcs]], domain_order_inv[domain_infos[dsts]])
        
        # set evaluator for the target scenario
        if self.metric is not None:
            if '@' in self.metric:
                metric_name, metric_k = self.metric.split('@')
                self.__evaluator = evaluator_map[metric_name](self.num_tasks, int(metric_k))
            else:
                self.__evaluator = evaluator_map[self.metric](self.num_tasks, self.__task_ids)
        self.__test_results = []
        
    def _update_target_dataset(self):
        # get sources and destinations
        srcs, dsts = self.__graph.edges()
        
        # note that the edges are bi-directed
        is_even = ((torch.arange(self.__inner_tvt_splits.shape[0]) % 2) == 0)
        
        # train/val/test - 8:1:1 random split
        edges_for_train = (self.__inner_tvt_splits < 8)
        if self.incr_type == 'time':
            edges_for_train &= (self.__task_ids <= self._curr_task)
        edges_ready = {'val': ((self.__inner_tvt_splits == 8) & (self.__task_ids == self._curr_task)) & is_even,
                       'test': (self.__inner_tvt_splits > 8) & is_even}
        
        # generate data using only train edges
        target_dataset = dgl.graph((srcs[edges_for_train], dsts[edges_for_train]), num_nodes=self.__graph.num_nodes())
        for k in self.__graph.ndata.keys():
            if (not self.minimize) or (k != 'time' or k != 'domain'): target_dataset.ndata[k] = self.__graph.ndata[k]
        for k in self.__graph.edata.keys():
            if (not self.minimize) or (k != 'time' or k != 'domain'): target_dataset.edata[k] = self.__graph.edata[k][edges_for_train]
            
        # prepare val/test data for current task (containing negative edges)
        target_edges = {_split: torch.stack((srcs[edges_ready[_split]], dsts[edges_ready[_split]]), dim=-1) for _split in ['val', 'test']}
        gt_labels = {_split: torch.cat((self.__task_ids[edges_ready[_split]] + 1,
                             torch.zeros(self.__neg_edges[_split].shape[0], dtype=torch.long)), dim=0) for _split in ['val', 'test']}
        randperms = {_split: torch.randperm(gt_labels[_split].shape[0]) for _split in ['val', 'test']}
        target_edges = {_split: torch.cat((target_edges[_split], self.__neg_edges[_split]), dim=0)[randperms[_split]] for _split in ['val', 'test']}

        # generate train/val/test dataset for current task
        edges_ready['train'] = (edges_for_train & is_even) & (self.__task_ids == self._curr_task)
        target_edges['train'] = torch.stack((srcs[edges_ready['train']], dsts[edges_ready['train']]), dim=-1)
        self.__target_labels = {_split: gt_labels[_split][randperms[_split]] for _split in ['val', 'test']}
        self._target_dataset = {'graph': dgl.add_self_loop(target_dataset),
                                'train': {'edge': target_edges['train']},
                                'val': {'edge': target_edges['val'], 'label': (self.__target_labels['val'] > 0).long()},
                                'test': {'edge': target_edges['test'], 'label': -torch.ones_like(self.__target_labels['test'])}}
        self._target_dataset['train']['label'] = torch.ones(self._target_dataset['train']['edge'].shape[0], dtype=torch.long)
        
    def _update_accumulated_dataset(self):
        # get sources and destinations
        srcs, dsts = self.__graph.edges()
        
        # note that the edges are bi-directed
        is_even = ((torch.arange(self.__inner_tvt_splits.shape[0]) % 2) == 0)
        
        # train/val/test - 8:1:1 random split
        edges_for_train = (self.__inner_tvt_splits < 8)
        if self.incr_type == 'time':
            edges_for_train &= (self.__task_ids <= self._curr_task)
        edges_ready = {'val': ((self.__inner_tvt_splits == 8) & (self.__task_ids <= self._curr_task)) & is_even,
                       'test': (self.__inner_tvt_splits > 8) & is_even}
        target_dataset = dgl.graph((srcs[edges_for_train], dsts[edges_for_train]), num_nodes=self.__graph.num_nodes())
        for k in self.__graph.ndata.keys():
            if (not self.minimize) or (k != 'time' or k != 'domain'): target_dataset.ndata[k] = self.__graph.ndata[k]
        for k in self.__graph.edata.keys():
            if (not self.minimize) or (k != 'time' or k != 'domain'): target_dataset.edata[k] = self.__graph.edata[k][edges_for_train]
            
        # prepare val/test data for current task (containing negative edges)
        target_edges = {_split: torch.stack((srcs[edges_ready[_split]], dsts[edges_ready[_split]]), dim=-1) for _split in ['val']}
        gt_labels = {_split: torch.cat((self.__task_ids[edges_ready[_split]] + 1,
                             torch.zeros(self.__neg_edges[_split].shape[0], dtype=torch.long)), dim=0) for _split in ['val']}

        randperms = {_split: torch.randperm(gt_labels[_split].shape[0]) for _split in ['val']}
        target_edges = {_split: torch.cat((target_edges[_split], self.__neg_edges[_split]), dim=0)[randperms[_split]] for _split in ['val']}
        
        # generate train/val/test dataset for current task
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
        """
            The inner function of get_eval_result.
            
            Args:
                preds (torch.Tensor): predicted output of the current model
                target_split (str): target split to measure the performance (spec., 'val' or 'test')
        """
        gt = (self.__target_labels[target_split] > 0).long()
        assert preds.shape == gt.shape, "shape mismatch"
        return self.__evaluator(preds, gt, self.__target_labels[target_split] - 1)
    
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
        gt = (self.__accumulated_labels[target_split] > 0).long()
        assert preds.shape == gt.shape, "shape mismatch"
        return self.__evaluator(preds, gt, self.__accumulated_labels[target_split] - 1)
        
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
    
class LCScenarioLoader(BaseScenarioLoader):
    """
        The sceanario loader for link classification.

        **Usage example:**

            >>> scenario = LCScenarioLoader(dataset_name="bitcoin", num_tasks=3, metric="accuracy", 
            ...             save_path="/data", incr_type="task", task_shuffle=True)
            
            >>> scenario = LCScenarioLoader(dataset_name="bitcoin", num_tasks=7, metric="aucroc", 
            ...             save_path="/data", incr_type="time")

        Bases: ``BaseScenarioLoader``
    """
    def _init_continual_scenario(self):
        self.num_feats, self.__graph = load_linkc_dataset(self.dataset_name, self.incr_type, self.save_path)
        self.num_classes = 1
        pkl_path = os.path.join(self.save_path, f'{self.dataset_name}_metadata_allIL.pkl')
        download(f'https://github.com/anonymous-submission-23/anonymous-submission-23.github.io/raw/main/_splits/{self.dataset_name}_metadata_allIL.pkl', pkl_path)
        metadata = pickle.load(open(pkl_path, 'rb'))
        self.__inner_tvt_splits = metadata['inner_tvt_split'] % 10
        
        if self.incr_type in ['domain']:
            raise NotImplementedError
        elif self.incr_type == 'time':
            # binary classification problem
            self.num_classes = 1
            self.num_tasks = 7
            # split into tasks using timestamp
            counts = torch.cumsum(torch.bincount(self.__graph.edata['time']), dim=-1)
            self.__task_ids = (counts / ((self.__graph.num_edges() + 1.) / self.num_tasks)).long()
            self.__task_ids = self.__task_ids[self.__graph.edata['time']]
            self.__graph.edata.pop('time')
            self.__graph.edata['label'] = (self.__graph.edata.pop('label') < 0).long()
        elif self.incr_type in ['class', 'task']:
            # multi-class classifcation problem
            self.label_to_class = torch.LongTensor([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 6, 6, 2, 3, 4, 5, 5, 5, 5, 5])
            self.num_classes = 7
            self.num_tasks = 3
            
            # determine task configuration
            if self.kwargs is not None and 'task_orders' in self.kwargs:
                self.__splits = tuple([torch.LongTensor(class_ids) for class_ids in self.kwargs['task_orders']])
            elif self.kwargs is not None and 'task_shuffle' in self.kwargs and self.kwargs['task_shuffle']:
                self.__splits = torch.split(torch.randperm(self.num_classes-1), self.num_classes // self.num_tasks)[:self.num_tasks]
            else:
                self.__splits = torch.split(torch.arange(self.num_classes-1), self.num_classes // self.num_tasks)[:self.num_tasks]
            
            # compute task ids for each instance and remove time information (since it is unnecessary)
            id_to_task = self.num_tasks * torch.ones(self.num_classes).long()
            for i in range(self.num_tasks):
                id_to_task[self.__splits[i]] = i
            self.__graph.edata['label'] = self.label_to_class[self.__graph.edata.pop('label').squeeze(-1) + 10]
            self.__graph.edata.pop('time')
            self.__task_ids = id_to_task[self.__graph.edata['label']]
        
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
        target_dataset = copy.deepcopy(self.__graph)
        
        # set train/val/test split for the current task
        target_dataset.edata['train_mask'] = (self.__inner_tvt_splits < 8) & (self.__task_ids == self._curr_task)
        target_dataset.edata['val_mask'] = (self.__inner_tvt_splits == 8) & (self.__task_ids == self._curr_task)
        target_dataset.edata['test_mask'] = (self.__inner_tvt_splits > 8)
        
        # hide labels of test nodes
        target_dataset.edata['label'] = self.__graph.edata['label'].clone()
        target_dataset.edata['label'][target_dataset.edata['test_mask'] | (self.__task_ids != self._curr_task)] = -1
        
        if self.incr_type == 'class':
            # for class-IL, no need to change
            self._target_dataset = target_dataset
        elif self.incr_type == 'task':
            # for task-IL, we need task information. BeGin provide the information with 'task_specific_mask'
            self._target_dataset = target_dataset
            self._target_dataset.edata['task_specific_mask'] = self.__task_masks[self.__task_ids]
        elif self.incr_type == 'time':
            # for time-IL, we need to hide unseen nodes and information at the current timestamp
            srcs, dsts = target_dataset.edges()
            edges_ready = (self.__task_ids <= self._curr_task)
            self._target_dataset = dgl.graph((srcs[edges_ready], dsts[edges_ready]), num_nodes=self.__graph.num_nodes())
            for k in target_dataset.ndata.keys():
                self._target_dataset.ndata[k] = target_dataset.ndata[k]
            for k in target_dataset.edata.keys():
                self._target_dataset.edata[k] = target_dataset.edata[k][edges_ready]
        elif self.incr_type == 'domain':
            pass
        
    def _update_accumulated_dataset(self):
        target_dataset = copy.deepcopy(self.__graph)
        
        # set train/val/test split for the current task
        target_dataset.edata['train_mask'] = (self.__inner_tvt_splits < 8) & (self.__task_ids <= self._curr_task)
        target_dataset.edata['val_mask'] = (self.__inner_tvt_splits == 8) & (self.__task_ids <= self._curr_task)
        target_dataset.edata['test_mask'] = (self.__inner_tvt_splits > 8)
        
        # hide labels of test nodes
        target_dataset.edata['label'] = self.__graph.edata['label'].clone()
        target_dataset.edata['label'][target_dataset.edata['test_mask'] | (self.__task_ids > self._curr_task)] = -1
        
        if self.incr_type == 'class':
            # for class-IL, no need to change
            self._accumulated_dataset = target_dataset
        elif self.incr_type == 'task':
            # for task-IL, we need task information. BeGin provide the information with 'task_specific_mask'
            self._accumulated_dataset = target_dataset
            self._accumulated_dataset.edata['task_specific_mask'] = self.__task_masks[self.__task_ids]
        elif self.incr_type == 'time':
            # for time-IL, we need to hide unseen nodes and information at the current timestamp
            srcs, dsts = target_dataset.edges()
            edges_ready = (self.__task_ids <= self._curr_task)
            self._accumulated_dataset = dgl.graph((srcs[edges_ready], dsts[edges_ready]), num_nodes=self.__graph.num_nodes())
            for k in target_dataset.ndata.keys():
                self._accumulated_dataset.ndata[k] = target_dataset.ndata[k]
            for k in target_dataset.edata.keys():
                self._accumulated_dataset.edata[k] = target_dataset.edata[k][edges_ready]
        elif self.incr_type == 'domain':
            pass
            
    def _get_eval_result_inner(self, preds, target_split):
        """
            The inner function of get_eval_result.
            
            Args:
                preds (torch.Tensor): predicted output of the current model
                target_split (str): target split to measure the performance (spec., 'val' or 'test')
        """
        if self.incr_type == 'time':
            # for Time-IL we evaluate the performance only with currently seen nodes
            gt = self.__graph.edata['label'][self.__task_ids <= self._curr_task][self._target_dataset.edata[target_split + '_mask']]
            assert preds.shape == gt.shape, "shape mismatch"
            return self.__evaluator(preds, gt, torch.arange(self.__graph.num_edges())[self.__task_ids <= self._curr_task][self._target_dataset.edata[target_split + '_mask']])
        else:
            gt = self.__graph.edata['label'][self._target_dataset.edata[target_split + '_mask']]
            assert preds.shape == gt.shape, "shape mismatch"
            return self.__evaluator(preds, gt, torch.arange(self._target_dataset.num_edges())[self._target_dataset.edata[target_split + '_mask']])
    
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
        if self.incr_type == 'time':
            # for Time-IL we evaluate the performance only with currently seen nodes
            gt = self.__graph.edata['label'][self.__task_ids <= self._curr_task][self._accumulated_dataset.edata[target_split + '_mask']]
            assert preds.shape == gt.shape, "shape mismatch"
            return self.__evaluator(preds, gt, torch.arange(self.__graph.num_edges())[self.__task_ids <= self._curr_task][self._accumulated_dataset.edata[target_split + '_mask']])
        else:
            gt = self.__graph.edata['label'][self._accumulated_dataset.edata[target_split + '_mask']]
            assert preds.shape == gt.shape, "shape mismatch"
            return self.__evaluator(preds, gt, torch.arange(self._target_dataset.num_edges())[self._accumulated_dataset.edata[target_split + '_mask']])
        
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