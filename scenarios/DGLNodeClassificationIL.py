from .DGLBasicIL import DGLBasicIL

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
    
    def get_simple_eval_result(self, curr_batch_preds, curr_batch_gts):
        return self.__evaluator.simple_eval(curr_batch_preds, curr_batch_gts)
    
    def next_task(self, preds=torch.empty(1)):
        self.__test_results.append(self._get_eval_result_inner(preds, target_split='test'))
        super().next_task(preds)
        if self._curr_task == self.num_tasks: return self.__test_results