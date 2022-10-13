from .DGLBasicIL import DGLBasicIL

class DGLGraphClassificationIL(DGLBasicIL):
    """ 
        aaaa
    """
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