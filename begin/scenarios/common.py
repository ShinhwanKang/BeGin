import torch

class BaseScenarioLoader:
    r"""Base framework for implementing scenario module.

    Arguments:
        dataset_name (str): The name of the dataset.
        save_path (str): The path where the dataset file is saved.
        num_tasks (int): The number of tasks in graph continual learning.
        incr_type (str): The incremental setting of graph continual learning (spec. task, class, domain, and time).
        metric (str): Basic metric to measure performance (spec., accuracy, AUCROC, and HITS@K).
        kwargs: (dict, optional): Keyword arguments to be passed to the scenario module (e.g., task_shuffle (bool): If true, fixed order, else random order)
    
    """
    def __init__(self, dataset_name, save_path, num_tasks, incr_type, metric, **kwargs):
        self.dataset_name = dataset_name
        self.save_path = save_path
        self.num_classes = None
        self.num_feats = None
        self.num_tasks = num_tasks
        self.incr_type = incr_type
        self.metric = metric
        self.kwargs = kwargs
        
        self._curr_task = 0
        self._target_dataset = None
        
        self._init_continual_scenario()
        self._update_target_dataset()
        self._update_accumulated_dataset()
        
    def _init_continual_scenario(self):
        """ 
            Load the entire dataset and initialize the setting of graph continual learning according the incremental setting.
        """
        raise NotImplementedError
    
    def _update_target_dataset(self):
        """ 
            Update the graph dataset the implemented model uses in the current task.
            According to the ``incr_type``, the information updated is different.

            Note:
                The implemented model can only process the training data in the current task.
        """
        raise NotImplementedError
    
    def _update_accumulated_dataset(self):
        """ 
            Update the graph dataset the joint model uses. 
            According to the ``incr_type``, the information updated is different.

            Note:
                The joint model can process all of training data in previous tasks including the current task.
        """
        raise NotImplementedError
    
    def __len__(self):
        """ 
            Returns:
                The number of tasks in this scenario
        """
        return self.num_tasks
    
    def next_task(self, preds):
        """ 
            Update graph datasets used in graph continual learning. 
            Specifically, the ``target`` denotes a dataset the implemented model uses and the ``accumulated`` denotes a dataset the joint model uses.

            Args:
                preds (torch.Tensor): Predicted output of the models
        """
        self._curr_task += 1
        if self._curr_task < self.num_tasks:
            self._update_target_dataset()
            self._update_accumulated_dataset()
            
    def get_current_dataset(self):
        """ 
            Returns:
                The graph dataset the implemented model uses in the current task
        """
        if self._curr_task >= self.num_tasks: return None
        return self._target_dataset
    
    def get_accumulated_dataset(self):
        """ 
            Returns:
                The graph dataset the joint model uses in the current task
        """
        if self._curr_task >= self.num_tasks: return None
        return self._accumulated_dataset