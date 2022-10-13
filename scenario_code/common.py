class DGLBasicIL:
    r"""Base framework for implementing scenario module

    Arguments:
        dataset_name (float): aa 
        save_path (float): aa 
        num_tasks (float): aa 
        incr_type (float): aa 
        cover_unseen (float): aa (DEFALUT : None)
        minimize (float): aa 
        metric (float): aa
        kwargs: (float): aa

    """
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device=None, **kwargs):
        """ 
            Initialize a base framework
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
            Returns
            -------- 
            the incremental setting (e.g., task, class, domain, and time) 
        """
        return self.__scenario.incr_type
    # def __init__(self, dataset_name=None, save_path='/mnt/d/graph_dataset', num_tasks=1, incr_type='class', cover_unseen=True, minimize=True, metric=None, **kwargs):
    #     """ 
    #         aaaadddd
    #     """
    #     self.dataset_name = dataset_name
    #     self.save_path = save_path
    #     self.num_classes = None
    #     self.num_feats = None
    #     self.num_tasks = num_tasks
    #     self.incr_type = incr_type
    #     self.cover_unseen = cover_unseen
    #     self.minimize = minimize
    #     self.metric = metric
    #     self.kwargs = kwargs
        
    #     self._curr_task = 0
    #     self._target_dataset = None
        
    #     self._init_continual_scenario()
        
    #     self._update_target_dataset()
    #     self._update_accumulated_dataset()
        
    # def _init_continual_scenario(self):
    #     raise NotImplementedError
    
    # def _update_target_dataset(self):
    #     raise NotImplementedError
    
    # def _update_accumulated_dataset(self):
    #     raise NotImplementedError
    
    # def __len__(self):
    #     return self.num_tasks
    
    # def next_task(self, preds=torch.empty(1)):
    #     """ 
    #         aaaa
    #     """
    #     self._curr_task += 1
    #     if self._curr_task < self.num_tasks:
    #         self._update_target_dataset()
    #         self._update_accumulated_dataset()
            
    # def get_current_dataset(self):
    #     if self._curr_task >= self.num_tasks: return None
    #     return self._target_dataset
    
    # def get_accumulated_dataset(self):
    #     if self._curr_task >= self.num_tasks: return None
    #     return self._accumulated_dataset