from .common import DGLBasicIL
import torch
import dgl

class DGLLinkPredictionIL(DGLBasicIL):
    """
        The sceanario loader for link prediction problems.

        **Usage example:**

            >>> scenario = DGLLinkPredictionIL(dataset_name="ogbl-collab", num_tasks=3, metric="hits@50", 
            ...             save_path="/data", incr_type="time", task_shuffle=True)

        Bases: ``DGLBasicIL``
    """
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
        
        target_dataset = dgl.graph((srcs[edges_for_train], dsts[edges_for_train]), num_nodes=self.__graph.num_nodes())
        for k in self.__graph.ndata.keys():
            if (not self.minimize) or (k != 'time' or k != 'domain'): target_dataset.ndata[k] = self.__graph.ndata[k]
        for k in self.__graph.edata.keys():
            if (not self.minimize) or (k != 'time' or k != 'domain'): target_dataset.edata[k] = self.__graph.edata[k][edges_for_train]

        target_edges = {_split: torch.stack((srcs[edges_ready[_split]], dsts[edges_ready[_split]]), dim=-1) for _split in ['val', 'test']}
        gt_labels = {_split: torch.cat((self.__task_ids[edges_ready[_split]] + 1,
                             torch.zeros(self.__neg_edges[_split].shape[0], dtype=torch.long)), dim=0) for _split in ['val', 'test']}
        

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

        
        target_dataset = dgl.graph((srcs[edges_for_train], dsts[edges_for_train]), num_nodes=self.__graph.num_nodes())
        for k in self.__graph.ndata.keys():
            if (not self.minimize) or (k != 'time' or k != 'domain'): target_dataset.ndata[k] = self.__graph.ndata[k]
        for k in self.__graph.edata.keys():
            if (not self.minimize) or (k != 'time' or k != 'domain'): target_dataset.edata[k] = self.__graph.edata[k][edges_for_train]
            
        target_edges = {_split: torch.stack((srcs[edges_ready[_split]], dsts[edges_ready[_split]]), dim=-1) for _split in ['val']}
        gt_labels = {_split: torch.cat((self.__task_ids[edges_ready[_split]] + 1,
                             torch.zeros(self.__neg_edges[_split].shape[0], dtype=torch.long)), dim=0) for _split in ['val']}

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