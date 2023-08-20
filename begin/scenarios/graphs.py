import torch
import dgl
import os
import pickle
import copy
from dgl.data.utils import download, Subset

from .common import BaseScenarioLoader
from .datasets import *
from . import evaluator_map

def load_graph_dataset(dataset_name, dataset_load_func, incr_type, save_path):
    domain_info, time_info = None, None
    if dataset_load_func is not None:
        custom_dataset = dataset_load_func(save_path=save_path)
        dataset = custom_dataset['graphs']
        num_feats = custom_dataset['num_feats']
        num_classes = custom_dataset['num_classes']
        domain_info = custom_dataset.get('domain_info', None)
        time_info = custom_dataset.get('time_info', None)
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
        dataset = DglGraphPropPredDataset('ogbg-molhiv', root=save_path)
        num_feats, num_classes = dataset[0][0].ndata['feat'].shape[-1], 1
        
        """ (For Task/Class-IL)
        # load train/val/test split
        split_idx = dataset.get_idx_split()
        for _split, _split_name in [('train', '_train'), ('valid', '_val'), ('test', '_test')]:
            _indices = torch.zeros(len(dataset), dtype=torch.bool)
            _indices[split_idx[_split]] = True
            setattr(dataset, _split_name + '_mask', _indices)
        """
        # load train/val/test split and domain_info
        pkl_path = os.path.join(save_path, f'molhivx_metadata_domainIL.pkl')
        download(f'https://github.com/anonymous-submission-23/anonymous-submission-23.github.io/raw/main/_splits/molhivx_metadata_domainIL.pkl', pkl_path)
        metadata = pickle.load(open(pkl_path, 'rb'))
        inner_tvt_splits = metadata['inner_tvt_splits']
        # set train/val/test split (random split, 8:1:1)
        dataset._train_masks = (inner_tvt_splits % 10) < 8
        dataset._val_masks = (inner_tvt_splits % 10) == 8
        dataset._test_masks = (inner_tvt_splits % 10) > 8
        domain_info = metadata['domain_splits']
        
    elif dataset_name in ['nyctaxi'] and incr_type in ['time']:
        dataset = NYCTaxiDataset(dataset_name, raw_dir=save_path)
        num_feats, num_classes = dataset[0][0].ndata['feat'].shape[-1], 2
        
        # load time split information and train/val/test splits (random split, 6:2:2)
        pkl_path = os.path.join(save_path, f'nyctaxi_metadata_timeIL.pkl')
        download(f'https://github.com/anonymous-submission-23/anonymous-submission-23.github.io/raw/main/_splits/nyctaxi_metadata_timeIL.pkl', pkl_path)
        metadata = pickle.load(open(pkl_path, 'rb'))
        inner_tvt_splits = metadata['inner_tvt_splits']
        dataset._train_masks = (inner_tvt_splits % 10) < 6
        dataset._val_masks = ((inner_tvt_splits % 10) == 6) | ((inner_tvt_splits % 10) == 7) 
        dataset._test_masks = (inner_tvt_splits % 10) > 7    
        time_info = metadata['time_splits']
    elif dataset_name in ['ogbg-ppa'] and incr_type in ['domain']:
        dataset = OgbgPpaSampledDataset(save_path)
        num_feats, num_classes = 2, 37
        pkl_path = os.path.join(save_path, f'ogbg-ppa_metadata_domainIL.pkl')
        download(f'https://github.com/jihoon-ko/BeGin/raw/main/metadata/ogbg-ppa_metadata_domainIL.pkl', path=pkl_path)
        metadata = pickle.load(open(pkl_path, 'rb'))
        inner_tvt_splits = metadata['inner_tvt_split']
        # set train/val/test split (random split, 8:1:1)
        dataset._train_masks = (inner_tvt_splits % 10) < 8
        dataset._val_masks = (inner_tvt_splits % 10) == 8
        dataset._test_masks = (inner_tvt_splits % 10) > 8
        domain_info = metadata['domain_info']
    elif dataset_name in ['sentiment'] and incr_type in ['time']:
        dataset = SentimentGraphDataset(dataset_name='sentiment', raw_dir=save_path)
        num_feats, num_classes = dataset._num_feats, dataset._num_classes
        time_info = dataset._time_info
        delattr(dataset, "_time_info")
    else:
        raise NotImplementedError("Tried to load unsupported scenario.")
    
    print("=====CHECK=====")
    print("num_classes:", num_classes, ", num_feats:", num_feats)
    print("dataset._train_mask:", dataset._train_masks.shape)
    print("dataset._val_mask:", dataset._val_masks.shape)
    print("dataset._test_mask:", dataset._test_masks.shape)
    print("dataset.labels:", dataset.labels.shape)
    if incr_type == 'time':
        print("time_info:", time_info is not None)
    if incr_type == 'domain':
        print("domain_info:", domain_info is not None)
    print("===============")
    
    return num_classes, num_feats, dataset, domain_info, time_info

class GCScenarioLoader(BaseScenarioLoader):
    """
        The sceanario loader for graph classification problems.

        **Usage example:**

            >>> scenario = GCScenarioLoader(dataset_name="ogbg-molhiv", num_tasks=10, metric="rocauc", 
            ...                             save_path="./data", incr_type="domain", task_shuffle=True)

        
        Bases: ``BaseScenarioLoader``
    """
    def _init_continual_scenario(self):
        self.num_classes, self.num_feats, self.__dataset, self.__domain_info, self.__time_splits = load_graph_dataset(self.dataset_name, self.dataset_load_func, self.incr_type, self.save_path)
        
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
            self.__task_ids = id_to_task[self.__dataset.labels]
            self.__original_labels = self.__dataset.labels.clone()
            self.__dataset.labels[self.__dataset._test_masks] = -1
        elif self.incr_type == 'time':
            # compute task ids for each instance
            self.num_tasks = self.__time_splits.max().item() + 1
            self.__task_ids = self.__time_splits
            self.__dataset.labels = self.__dataset.labels.squeeze()
            self.__original_labels = self.__dataset.labels.clone()
            self.__dataset.labels[self.__dataset._test_masks] = -1
        elif self.incr_type == 'domain':
            # determine task configuration
            self.num_tasks = self.__domain_info.max().item() + 1
            if self.kwargs is not None and 'task_shuffle' in self.kwargs and self.kwargs['task_shuffle']:
                self.__task_order = torch.randperm(self.num_tasks)
                print('domain_order:', self.__task_order)
                self.__task_ids = self.__task_order[self.__domain_info]
            else:
                self.__task_ids = self.__domain_info
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
        if _global:
            metadata = {'num_classes': self.num_classes, 'task': self.__task_ids}
            if _global and self.incr_type == 'task':  metadata['task_specific_mask'] = self.__task_masks[self.__task_ids]
            metadata['graphs'] = []
            metadata['labels'] = []
            for i in tqdm.trange(len(self.__dataset)):
                if self.incr_type != 'task':
                    g, l = self.__dataset[i]
                else:
                    g, l, _ = self.__dataset[i]
                g_data = {}
                g_data['edges'] = g.edges()
                g_data['ndata_feat'] = g.ndata['feat']
                if 'feat' in g.edata: g_data['edata_feat'] = g.edata['feat']
                metadata['graphs'].append(g_data)
                metadata['labels'].append(l)
            metadata['labels'] = torch.LongTensor(metadata['labels'])
            metadata['train_mask'] = self.__dataset._train_masks
            metadata['val_mask'] = self.__dataset._val_masks
            metadata['test_mask'] = self.__dataset._test_masks
            metadata['test_indices'] = torch.nonzero(self.__dataset._test_masks, as_tuple=True)[0]
        else:
            metadata = {}
            metadata['train_indices'] = torch.nonzero((self.__task_ids == self._curr_task) & self.__dataset._train_masks, as_tuple=True)[0]
            metadata['val_indices'] = torch.nonzero((self.__task_ids == self._curr_task) & self.__dataset._val_masks, as_tuple=True)[0]
        return metadata