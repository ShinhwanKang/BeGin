import torch
import tqdm

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
        self.dataset_load_func = kwargs.get('dataset_load_func', None)
        self.save_path = save_path
        self.num_classes = None
        self.num_feats = None
        self.num_tasks = num_tasks
        self.incr_type = incr_type
        self.metric = metric
        self.kwargs = kwargs
        
        self._curr_task = 0
        self._target_dataset = None
        self.initial_test_result = None
        self.export_mode = False
        
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
    
    def get_current_dataset_for_export(self, _global=False):
        return None
    
    def export_dataset(self, full=True):
        """
            Export the continual learning scenarios.
            We provide this functionality for flexibility, by providing the datasets separately.
            Depending on the target problem, the output format varies. Detailed format information is as follows:
            
            (1) For NC, the global information contains number of classes (``num_classes``), node features (``ndata_feat``), task id for each node (``task``), edges of the whole graph (``edges``), mask for data splits (``train_mask``, ``val_mask``, and ``test_mask``), and ground-truth labels (``label``).
            For each task, the task-specific information contains edges of the graph observed at the current task (``edges``), mask for train/validation splits (``train_mask``, ``val_mask``), and the ground-truth labels observed at the current task (``label``).
            
            (2) For LC, the global information contains number of classes (``num_classes``), node features (``ndata_feat``), task id for each edge (``task``), edges of the whole graph (``edges``), mask for data splits (``train_mask``, ``val_mask``, and ``test_mask``), and ground-truth labels (``label``).
            For each task, the task-specific information contains edges of the graph observed at the current task (``edges``), mask for train/validation splits (``train_mask``, ``val_mask``), and the ground-truth labels observed at the current task (``label``).
            
            (3) For LP, the global information contains node features (``ndata_feat``), task id for each edge (``task``), edges of the whole graph (``edges``), negative edges for evaluation (``neg_edges``), edges containing test edges and negative edges for evaluation (``test_edges``), and their corresponding labels (``test_label``). The test label is ``l`` is 0 if it is negative edge, otherwise it is ground-truth edge for task ``l``.
            For each task, the task-specific information contains base edges observed at the current task (``edges``), edges for training prediction problem and their corresponding labels (``val_edges``, ``val_label``), and edges for validation and their corresponding labels  (``test_edges``, ``test_label``).
            
            (4) For GC, the global information contains number of classes (``num_classes``), node features (``ndata_feat``), task id for each graph (``task``), graphs of the whole dataset (``graphs``), mask for data splits (``train_mask``, ``val_mask``, and ``test_mask``), indices of test graphs (``test_indices``) and ground-truth labels (``label``).
            For each task, the task-specific information contains indices of training graphs (``train_indices``) and indices of validation graphs (``val_indices``).
            
            Args:
                full (boolean, Optional): if ``full=True``, the returned exported dataset contains both global information (``output['global']``) and task-specific information (``output['tasks']``).
                Otherwise, the returned exported dataset contains only global infomation. 
                
            Returns:
                The exported scenario (dict).
        """
        exported_data = {'global': self.get_current_dataset_for_export(_global=True)}
        if full:
            self.export_mode = True
            if self._curr_task != 0:
                print("[ERROR] you can call this function iff current task is the first task!")
                raise NotImplementedError
            exported_data['tasks'] = {}
            for i in tqdm.trange(self.num_tasks):
                exported_data['tasks'][self._curr_task] = self.get_current_dataset_for_export(_global=False)
                self.next_task(None)
                
        return exported_data