import torch
import dgl
import os
import pickle
import copy
from dgl.data.utils import download, Subset

from .common import BaseScenarioLoader
from .datasets import *
from . import evaluator_map

def load_graph_dataset(dataset_name, incr_type, save_path):
    if dataset_name in ['mnist', 'cifar10'] and incr_type in ['task', 'class']:
        dataset = DGLGNNBenchmarkDataset(dataset_name, raw_dir=save_path)
        num_feats, num_classes = dataset.num_feats, dataset.num_classes
    elif dataset_name in ['aromaticity'] and incr_type in ['task', 'class']:
        dataset = AromaticityDataset(raw_dir=save_path)
        num_feats, num_classes = 2, 30
        # load train/val/test split (6:2:2 random split)
        pkl_path = os.path.join(save_path, f'{dataset_name}_metadata_allIL.pkl')
        download(f'https://github.com/anonymous-submission-23/anonymous-submission-23.github.io/raw/main/_splits/{dataset_name}_metadata_allIL.pkl', pkl_path)
        metadata = pickle.load(open(pkl_path, 'rb'))
        inner_tvt_splits = metadata['inner_tvt_splits']
        dataset._train_masks = (inner_tvt_splits % 10) < 6
        dataset._val_masks = ((inner_tvt_splits % 10) == 6) | ((inner_tvt_splits % 10) == 7) 
        dataset._test_masks = (inner_tvt_splits % 10) > 7
    elif dataset_name in ['ogbg-molhiv'] and incr_type in ['domain']:
        dataset = DglGraphPropPredDataset(dataset_name, root=save_path)
        num_feats, num_classes = dataset[0][0].ndata['feat'].shape[-1], 1
        # load train/val/test split
        split_idx = dataset.get_idx_split()
        for _split, _split_name in [('train', '_train'), ('valid', '_val'), ('test', '_test')]:
            _indices = torch.zeros(len(dataset), dtype=torch.bool)
            _indices[split_idx[_split]] = True
            setattr(dataset, _split_name + '_mask', _indices)
    elif dataset_name in ['nyctaxi'] and incr_type in ['time']:
        dataset = NYCTaxiDataset(dataset_name, raw_dir=save_path)
        num_feats, num_classes = dataset[0][0].ndata['feat'].shape[-1], 2
    else:
        raise NotImplementedError("Tried to load unsupported scenario.")
        
    return num_classes, num_feats, dataset

class GCScenarioLoader(BaseScenarioLoader):
    """
        The sceanario loader for graph classification problems.

        **Usage example:**

            >>> scenario = GCScenarioLoader(dataset_name="ogbg-molhiv", num_tasks=10, metric="rocauc", 
            ...                             save_path="./data", incr_type="domain", task_shuffle=True)

        
        Bases: ``BaseScenarioLoader``
    """
    def _init_continual_scenario(self):
        self.num_classes, self.num_feats, self.__dataset = load_graph_dataset(self.dataset_name, self.incr_type, self.save_path)
        
        if self.incr_type in ['class', 'task']:
            # determine task configuration
            if self.kwargs is not None and 'task_orders' in self.kwargs:
                self.__splits = tuple([torch.LongTensor(class_ids) for class_ids in self.kwargs['task_orders']])
            elif self.kwargs is not None and 'task_shuffle' in self.kwargs and self.kwargs['task_shuffle']:
                self.__splits = torch.split(torch.randperm(self.num_classes), self.num_classes // self.num_tasks)[:self.num_tasks]
            else:
                self.__splits = torch.split(torch.arange(self.num_classes), self.num_classes // self.num_tasks)[:self.num_tasks]
            
            # compute task ids for each instance
            print('class split information:', self.__splits)
            id_to_task = self.num_tasks * torch.ones(self.num_classes).long()
            for i in range(self.num_tasks):
                id_to_task[self.__splits[i]] = i
            self.__task_ids = id_to_task[self.__dataset._labels]
            self.__original_labels = self.__dataset._labels.clone()
            self.__dataset._labels[self.__dataset._test_masks] = -1
        elif self.incr_type == 'time':
            # load time split information and train/val/test splits (random split, 6:2:2)
            pkl_path = os.path.join(self.save_path, f'{self.dataset_name}_metadata_timeIL.pkl')
            download(f'https://github.com/anonymous-submission-23/anonymous-submission-23.github.io/raw/main/_splits/{self.dataset_name}_metadata_timeIL.pkl', pkl_path)
            metadata = pickle.load(open(pkl_path, 'rb'))
            inner_tvt_splits = metadata['inner_tvt_splits']
            self.__time_splits = metadata['time_splits']
            self.__dataset._train_masks = (inner_tvt_splits % 10) < 6
            self.__dataset._val_masks = ((inner_tvt_splits % 10) == 6) | ((inner_tvt_splits % 10) == 7) 
            self.__dataset._test_masks = (inner_tvt_splits % 10) > 7
            
            # compute task ids for each instance
            self.num_tasks = self.__time_splits.max().item() + 1
            self.__task_ids = self.__time_splits
            self.__dataset._labels = self.__dataset._labels.squeeze()
            self.__original_labels = self.__dataset._labels.clone()
            self.__dataset._labels[self.__dataset._test_masks] = -1
        elif self.incr_type == 'domain':
            # load domain information and train/val/test splits
            pkl_path = os.path.join(self.save_path, f'{self.dataset_name}_metadata_domainIL.pkl')
            download(f'https://github.com/anonymous-submission-23/anonymous-submission-23.github.io/raw/main/_splits/{self.dataset_name}_metadata_domainIL.pkl', pkl_path)
            metadata = pickle.load(open(pkl_path, 'rb'))
            inner_tvt_splits = metadata['inner_tvt_splits']
            domain_info = metadata['domain_splits']
            
            # determine task configuration
            self.num_tasks = domain_info.max().item() + 1
            if self.kwargs is not None and 'task_shuffle' in self.kwargs and self.kwargs['task_shuffle']:
                self.__task_order = torch.randperm(self.num_tasks)
                print('domain_order:', self.__task_order)
                self.__task_ids = self.__task_order[domain_info]
            else:
                self.__task_ids = domain_info
            
            # set train/val/test split (random split, 8:1:1)
            self.__dataset._train_masks = (inner_tvt_splits % 10) < 8
            self.__dataset._val_masks = (inner_tvt_splits % 10) == 8
            self.__dataset._test_masks = (inner_tvt_splits % 10) > 8
            self.__original_labels = self.__dataset.labels.clone()
            self.__dataset.labels[self.__dataset._test_masks] = -1
            
        # we need to provide task information (only for task-IL)
        if self.incr_type == 'task':
            self.__task_masks = torch.zeros(self.num_tasks + 1, self.num_classes).bool()
            for i in range(self.num_tasks):
                self.__task_masks[i, self.__splits[i]] = True
            self.__dataset._task_specific_masks = self.__task_masks[self.__task_ids]
        
        # set evaluator for the target scenario
        if self.metric is not None:
            self.__evaluator = evaluator_map[self.metric](self.num_tasks, self.__task_ids)
        self.__test_results = []
        
    def _update_target_dataset(self):
        # create subset for training / validation / test of the current task
        target_train_indices = torch.nonzero((self.__task_ids == self._curr_task) & self.__dataset._train_masks, as_tuple=True)[0]
        target_val_indices = torch.nonzero((self.__task_ids == self._curr_task) & self.__dataset._val_masks, as_tuple=True)[0]
        target_test_indices = torch.nonzero(self.__dataset._test_masks, as_tuple=True)[0]
        self._target_dataset = {'train': Subset(self.__dataset, target_train_indices), 'val': Subset(self.__dataset, target_val_indices), 'test': Subset(self.__dataset, target_test_indices)}
        
    def _update_accumulated_dataset(self):
        # create accumulated subset for training / validation / test of the current task
        target_train_indices = torch.nonzero((self.__task_ids <= self._curr_task) & self.__dataset._train_masks, as_tuple=True)[0]
        target_val_indices = torch.nonzero((self.__task_ids <= self._curr_task) & self.__dataset._val_masks, as_tuple=True)[0]
        target_test_indices = torch.nonzero(self.__dataset._test_masks, as_tuple=True)[0]
        self._accumulated_dataset = {'train': Subset(self.__dataset, target_train_indices), 'val': Subset(self.__dataset, target_val_indices), 'test': Subset(self.__dataset, target_test_indices)}
        
    def _get_eval_result_inner(self, preds, target_split):
        """
            The inner function of get_eval_result.
            
            Args:
                preds (torch.Tensor): predicted output of the current model
                target_split (str): target split to measure the performance (spec., 'val' or 'test')
        """
        gt = self.__original_labels[self._target_dataset[target_split].indices]
        assert preds.shape == gt.shape, "shape mismatch"
        return self.__evaluator(preds, gt, self._target_dataset[target_split].indices)
    
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
        gt = self.__original_labels[self._accumulated_dataset[target_split].indices]
        assert preds.shape == gt.shape, "shape mismatch"
        return self.__evaluator(preds, gt, self._accumulated_dataset[target_split].indices)
    
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