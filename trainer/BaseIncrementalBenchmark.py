from .BaseContinualFramework import BaseContinualFramework

class BaseIncrementalBenchmark(BaseContinualFramework):
    """
    Base framework for graph continual learning111

    :param int model: ...
    :param int scenario: ...
    :param int optimizer_fn: ...
    :param int loss_fn: ...
    :param int,optional device: ...
    :param int,optional kwargs: ...
    """
    def __init__(self, model, scenario, optimizer_fn, loss_fn=None, device=None, **kwargs):
        """
        :returns int: a+b
        """
        self.args = kwargs['args']
        if self.args.benchmark:
            dgl.seed(self.args.seed)
            dgl.random.seed(self.args.seed)
            
        self.__scenario = scenario
        self.__timestamp = str(time.time()).replace('.', '')
        self.__model = model
        self.__optimizer = optimizer_fn(self.__model.parameters())
        
        self.tmp_path = kwargs.get('tmp_save_path', 'results/22-graph-continual/tmp')
        self.result_path = kwargs.get('tmp_save_path', 'results/22-graph-continual/results')
        self.save_file_name = f'{self.args.dataset_name}_{self.__scenario.incr_type}_{self.args.seed}_{self.args.shuffle}_{self.args.algo_type}_{self.args.lr}_{self.args.weight_decay}_{self.args.dropout_rate}_{self.args.num_layers}_{self.args.hidden_dim}'
        
        self.__model_weight_path = f'{self.tmp_path}/{self.save_file_name}_model.pkt'
        self.__optim_weight_path = f'{self.tmp_path}/{self.save_file_name}_optimizer.pkt'
        # self.__model_weight_path = f'tmp/tmp_model.pkt'
        # self.__optim_weight_path = f'tmp/tmp_optimizer.pkt'
        torch.save(self.__model.state_dict(), self.__model_weight_path)
        torch.save(self.__optimizer.state_dict(), self.__optim_weight_path)
        
        self.__base_model = copy.deepcopy(model)
        self.__base_optimizer = optimizer_fn(self.__base_model.parameters())
        self._reset_model(self.__base_model)
        self._reset_optimizer(self.__base_optimizer)
        
        self.__accum_model = copy.deepcopy(model)
        self.__accum_optimizer = optimizer_fn(self.__accum_model.parameters())
        self._reset_model(self.__accum_model)
        self._reset_optimizer(self.__accum_optimizer)
    
        self.loss_fn = loss_fn if loss_fn is not None else (lambda x: None)
        self.device = device
        
        self.num_tasks = scenario.num_tasks
        self.eval_fn = lambda x, y: scenario.get_simple_eval_result(x, y)
        self.full_mode = kwargs.get('full_mode', False)

    @property
    def incr_type(self):
        """
        :returns int: a+b
        """
        return self.__scenario.incr_type
        
    @property
    def curr_task(self):
        """
        :returns int: a+b
        """
        return self.__scenario._curr_task
    
    def _reset_model(self, target_model):
        """
        :returns int: a+b
        """
        target_model.load_state_dict(torch.load(self.__model_weight_path))
        
    def _reset_optimizer(self, target_optimizer):
        target_optimizer.load_state_dict(torch.load(self.__optim_weight_path))
        
    def fit(self, epoch_per_task = 1):
        """
        :returns int: a+b
        """
        base_eval_results = {'base': {'val': [], 'test': []}, 'accum': {'val': [], 'test': []}, 'exp': {'val': [], 'test': []}}
        initial_training_state = self._initTrainingStates(self.__scenario, self.__model, self.__optimizer)
        training_states = {'exp': copy.deepcopy(initial_training_state), 'base': None, 'accum': None}
        initial_results = {'val': None, 'test': None}
        
        while True:
            curr_dataset = self.__scenario.get_current_dataset()
            accumulated_dataset = self.__scenario.get_accumulated_dataset()
            
            if curr_dataset is None: # overall training is done!
                break
            
            # re-initialize base (control) experiment
            training_states['base'] = copy.deepcopy(initial_training_state)
            self._reset_model(self.__base_model)
            self._reset_optimizer(self.__base_optimizer)
            
            training_states['accum'] = copy.deepcopy(initial_training_state)
            self._reset_model(self.__accum_model)
            self._reset_optimizer(self.__accum_optimizer)
            
            models = {'exp': self.__model, 'base': self.__base_model, 'accum': self.__accum_model}
            optims = {'exp': self.__optimizer, 'base': self.__base_optimizer, 'accum': self.__accum_optimizer}
            stop_training = {'exp': False, 'base': False, 'accum': False}
            dataloaders = {}
            for exp_name in ['exp', 'base']:
                dataloaders[exp_name] = {k: v for k, v in zip(['train', 'val', 'test'], self.prepareLoader(curr_dataset, training_states[exp_name]))}
                self._processBeforeTraining(self.__scenario._curr_task, curr_dataset, models[exp_name], optims[exp_name], training_states[exp_name])
            
            dataloaders['accum'] = {k: v for k, v in zip(['train', 'val', 'test'], self.prepareLoader(accumulated_dataset, training_states['accum']))}
            self._processBeforeTraining(self.__scenario._curr_task, accumulated_dataset, models['accum'], optims['accum'], training_states['accum'])    
            
            if self.__scenario._curr_task == 0:
                with torch.no_grad():
                    self.__base_model.eval()
                    curr_observed_mask = self.__base_model.classifier.observed.clone()
                    self.__base_model.classifier.observed.fill_(True)
                    for split in ['val', 'test']:
                        initial_stats = {}
                        initial_test_predictions = []
                        for curr_batch in iter(dataloaders['base'][split]):
                            initial_test_predictions.append(self.__evalWrapper(models['base'], curr_batch, initial_stats))
                        initial_results[split] = self.__scenario.get_eval_result(torch.cat(initial_test_predictions, dim=0), target_split=split)
                    print(initial_results)
                    self.__base_model.classifier.observed.copy_(curr_observed_mask)
                    
            for epoch_cnt in range(epoch_per_task):
                for exp_name in ['exp', 'base', 'accum'] if self.full_mode else ['exp']:
                    if stop_training[exp_name]: continue
                    train_stats = {}
                    models[exp_name].train()
                    for curr_batch in iter(dataloaders[exp_name]['train']):
                        self.__trainWrapper(models[exp_name], optims[exp_name], curr_batch, training_states[exp_name], train_stats)
                    reduced_train_stats = self._reduceTrainingStats(train_stats)
                
                    models[exp_name].eval()
                    val_stats, val_predictions = {}, []
                    for curr_batch in iter(dataloaders[exp_name]['val']):
                        val_predictions.append(self.__evalWrapper(models[exp_name], curr_batch, val_stats))
                    reduced_val_stats = self._reduceEvalStats(val_stats)
                    if exp_name == 'accum':
                        val_metric_result = self.__scenario.get_accum_eval_result(torch.cat(val_predictions, dim=0), target_split='val')[-1].item()
                    else:
                        val_metric_result = self.__scenario.get_eval_result(torch.cat(val_predictions, dim=0), target_split='val')[self.__scenario._curr_task].item()
                    
                    if exp_name == 'exp':
                        self._processTrainingLogs(self.__scenario._curr_task, epoch_cnt, val_metric_result, reduced_train_stats, reduced_val_stats)
                    
                    curr_iter_results = {'val_metric': val_metric_result, 'train_stats': reduced_train_stats, 'val_stats': reduced_val_stats}
                    if not self._processAfterEachIteration(models[exp_name], optims[exp_name], training_states[exp_name], curr_iter_results):
                        stop_training[exp_name] = True
                        sys.stderr.write(exp_name + ": TRAINING_STOPPED\n")
            
            for exp_name in ['base', 'accum', 'exp'] if self.full_mode else ['exp']:
                models[exp_name].eval()
                self._processAfterTraining(self.__scenario._curr_task, curr_dataset, models[exp_name], optims[exp_name], training_states[exp_name])
                    
            for split in ['val', 'test']:
                for exp_name in ['base', 'accum', 'exp'] if self.full_mode else ['exp']:
                    models[exp_name].eval()
                    test_predictions, test_stats = [], {}
                    for curr_batch in iter(dataloaders['accum'][split]):
                        test_predictions.append(self.__evalWrapper(models[exp_name], curr_batch, test_stats))

                    test_predictions = torch.cat(test_predictions, dim=0)
                    if exp_name == 'exp' and split == 'test':
                        eval_results = self.__scenario.next_task(test_predictions)
                    elif split == 'val':
                        base_eval_results[exp_name][split].append(self.__scenario.get_accum_eval_result(test_predictions, target_split=split))
                    else:
                        base_eval_results[exp_name][split].append(self.__scenario.get_eval_result(test_predictions, target_split=split))
                
        if self.full_mode:
            return {'init_val': initial_results['val'],
                    'init_test': initial_results['test'],
                    'exp_val': torch.stack(base_eval_results['exp']['val'], dim=0),
                    'exp_test': torch.stack(eval_results, dim=0),
                    'base_val': torch.stack(base_eval_results['base']['val'], dim=0),
                    'base_test': torch.stack(base_eval_results['base']['test'], dim=0),
                    'accum_val': torch.stack(base_eval_results['accum']['val'], dim=0),
                    'accum_test': torch.stack(base_eval_results['accum']['test'], dim=0),
                   }
        else:
            return {'init_val': initial_results['val'],
                    'init_test': initial_results['test'],
                    'exp_val': torch.stack(base_eval_results['exp']['val'], dim=0),
                    'exp_test': torch.stack(eval_results, dim=0),
                   }
    
    def _initTrainingStates(self, dataset, model, optimizer):
        return {}
    
    def prepareLoader(self, curr_dataset, curr_training_states):
        raise NotImplementedError
        
    def _processBeforeTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        pass
    
    def processTrainIteration(self, model, optimizer, curr_batch, training_states):
        raise NotImplementedError
    
    def __trainWrapper(self, model, optimizer, curr_batch, training_states, curr_stats):
        new_stats = self.processTrainIteration(model, optimizer, curr_batch, training_states)
        if new_stats is not None:
            for k, v in new_stats.items():
                if k not in curr_stats:
                    curr_stats[k] = []
                curr_stats[k].append(v)
                
    def _reduceTrainingStats(self, curr_stats):
        if '_num_items' not in curr_stats:
            reduced_stats = {k: sum(v) / len(v) for k, v in curr_stats.items()}
        else:
            weights = np.array(curr_stats.pop('_num_items'))
            total = weights.sum()
            reduced_stats = {k: (np.array(v) * weights).sum() / total for k, v in curr_stats.items()}
        return reduced_stats
    
    def processEvalIteration(self, model, curr_batch):
        raise NotImplementedError
    
    def __evalWrapper(self, model, curr_batch, curr_stats):
        preds, new_stats = self.processEvalIteration(model, curr_batch)
        if new_stats is not None:
            for k, v in new_stats.items():
                if k not in curr_stats:
                    curr_stats[k] = []
                curr_stats[k].append(v)
        return preds
    
    def _reduceEvalStats(self, curr_stats):
        if '_num_items' not in curr_stats:
            reduced_stats = {k: sum(v) / len(v) for k, v in curr_stats.items()}
        else:
            weights = np.array(curr_stats.pop('_num_items'))
            total = weights.sum()
            reduced_stats = {k: (np.array(v) * weights).sum() / total for k, v in curr_stats.items()}
        return reduced_stats
    
    def _processTrainingLogs(self, task_id, epoch_cnt, val_metric_result, train_stats, val_stats):
        pass
    
    def _processAfterEachIteration(self, curr_model, curr_optimizer, curr_training_states, curr_iter_results):
        return True
    
    def _processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        pass