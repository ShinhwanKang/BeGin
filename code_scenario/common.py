import torch

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
    def __init__(self, dataset_name=None, save_path='/mnt/d/graph_dataset', num_tasks=1, incr_type='class', cover_unseen=True, minimize=True, metric=None, **kwargs):
        """ 
            aaaadddd
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
        """ 
            aaaadddd
        """
        raise NotImplementedError
    
    def _update_target_dataset(self):
        """ 
            aaaadddd
        """
        raise NotImplementedError
    
    def _update_accumulated_dataset(self):
        """ 
            aaaadddd
        """
        raise NotImplementedError
    
    def __len__(self):
        """ 
            Return the number of tasks
        """
        return self.num_tasks
    
    def next_task(self, preds=torch.empty(1)):
        """ 
            Return data of the next task
        """
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