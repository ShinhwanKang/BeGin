import time
import copy
import torch
from torch import nn
import numpy as np
import sys
import dgl

class BaseTrainer:
    r""" Base framework for implementing trainer module.

    Arguments:
        model (torch.nn.Module): Pytorch model for graph continual learning.
        scenario (ScenarioLoader): The scenario module.
        optimizer_fn (lambda x: torch.optim.Optimizer): A generator function for optimizer.
        loss_fn : A loss function.
        device (str): target GPU device.
        kwargs (dict, optional): Keyword arguments to be passed to the trainer module.

    Note:
        For instance, by kwargs, users can pass hyperparameters the implemented method needs or a scheduler function (torch.nn) for tranining.  
    """
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device=None, **kwargs):
        # set random seed for DGL
        if kwargs.get('benchmark', False):
            fixed_seed = kwargs.get('seed', 0)
            dgl.seed(fixed_seed)
            dgl.random.seed(fixed_seed)
            
        self.__scenario = scenario
        self.__timestamp = str(time.time()).replace('.', '')
        self.__model = model
        self.__optimizer = optimizer_fn(self.__model.parameters())
        
        # set path for storing initial parameters of model and optimizer
        self.tmp_path = kwargs.get('tmp_save_path', 'tmp')
        self.result_path = kwargs.get('tmp_save_path', 'results')
        self.__model_weight_path = f'{self.tmp_path}/init_model.pkt'
        self.__optim_weight_path = f'{self.tmp_path}/init_optimizer.pkt'
        
        torch.save(self.__model.state_dict(), self.__model_weight_path)
        torch.save(self.__optimizer.state_dict(), self.__optim_weight_path)
        
        # initial settings for base model and joint model (they are used only when full_mode is enabled)
        self.__base_model = copy.deepcopy(model)
        self.__base_optimizer = optimizer_fn(self.__base_model.parameters())
        self._reset_model(self.__base_model)
        self._reset_optimizer(self.__base_optimizer)
        self.__accum_model = copy.deepcopy(model)
        self.__accum_optimizer = optimizer_fn(self.__accum_model.parameters())
        self._reset_model(self.__accum_model)
        self._reset_optimizer(self.__accum_optimizer)
        
        # other initialization settings 
        self.loss_fn = loss_fn if loss_fn is not None else (lambda x: None) # loss function
        self.device = device # gpu device
        self.num_tasks = scenario.num_tasks # number of tasks
        self.eval_fn = lambda x, y: scenario.get_simple_eval_result(x, y) # evaluation function for minibatches
        self.full_mode = kwargs.get('full_mode', False) # base model and joint model are used when full_mode=True

    @property
    def incr_type(self):
        """ 
            Returns:
                The incremental setting (e.g., task, class, domain, or time).
        """
        return self.__scenario.incr_type
        
    @property
    def curr_task(self):
        """ 
            Returns: 
                The id of a current task :math:`[0,T-1]`.
        """
        return self.__scenario._curr_task
    
    def _reset_model(self, target_model):
        """ 
            Reinitialize a model.
        """
        target_model.load_state_dict(torch.load(self.__model_weight_path))
        
    def _reset_optimizer(self, target_optimizer):
        """ 
            Reinitialize an optimizer.
        """
        target_optimizer.load_state_dict(torch.load(self.__optim_weight_path))
        
    def run(self, epoch_per_task = 1):
        """
            Run the overall process of graph continual learning optimization. 
        """
        # dictionary to store evaluation results
        base_eval_results = {'base': {'val': [], 'test': []}, 'accum': {'val': [], 'test': []}, 'exp': {'val': [], 'test': []}}
        # variable for initialized training state
        initial_training_state = self.initTrainingStates(self.__scenario, self.__model, self.__optimizer)
        # dictionary to store training states of the models
        training_states = {'exp': copy.deepcopy(initial_training_state), 'base': None, 'accum': None}
        # dictionary to store initial performances
        initial_results = {'val': None, 'test': None}
        
        while True:
            # load dataset for the current task and the accumulated dataset until the current task
            curr_dataset = self.__scenario.get_current_dataset()
            accumulated_dataset = self.__scenario.get_accumulated_dataset()
            
            if curr_dataset is None: # overall training is done!
                break
            
            # re-initialize base model and joint model (at the every beginning of training)
            training_states['base'] = copy.deepcopy(initial_training_state)
            self._reset_model(self.__base_model)
            self._reset_optimizer(self.__base_optimizer)
            training_states['accum'] = copy.deepcopy(initial_training_state)
            self._reset_model(self.__accum_model)
            self._reset_optimizer(self.__accum_optimizer)
            
            # dictionaries to store current models and optimizers
            models = {'exp': self.__model, 'base': self.__base_model, 'accum': self.__accum_model}
            optims = {'exp': self.__optimizer, 'base': self.__base_optimizer, 'accum': self.__accum_optimizer}
            
            # dictionary to store whether we need to stop training or not 
            stop_training = {'exp': False, 'base': False, 'accum': False}
            
            # dictionary to store dataloader for each model
            # after preparing dataloader, it calls 'processBeforeTraining' event function 
            dataloaders = {}
            for exp_name in ['exp', 'base']:
                dataloaders[exp_name] = {k: v for k, v in zip(['train', 'val', 'test'], self.prepareLoader(curr_dataset, training_states[exp_name]))}
                self.processBeforeTraining(self.__scenario._curr_task, curr_dataset, models[exp_name], optims[exp_name], training_states[exp_name])
            dataloaders['accum'] = {k: v for k, v in zip(['train', 'val', 'test'], self.prepareLoader(accumulated_dataset, training_states['accum']))}
            self.processBeforeTraining(self.__scenario._curr_task, accumulated_dataset, models['accum'], optims['accum'], training_states['accum'])    
            
            # compute initial performance
            if self.curr_task == 0:
                with torch.no_grad():
                    self.__base_model.eval()
                    curr_observed_mask = self.__base_model.classifier.observed.clone()
                    # to enable to predict all classes
                    self.__base_model.classifier.observed.fill_(True)
                    for split in ['val', 'test']:
                        initial_stats = {}
                        initial_test_predictions = []
                        # collect predicted results on current val/test data
                        for curr_batch in iter(dataloaders['base'][split]):
                            initial_test_predictions.append(self._evalWrapper(models['base'], curr_batch, initial_stats))
                        # compute the initial performances
                        initial_results[split] = self.__scenario.get_eval_result(torch.cat(initial_test_predictions, dim=0), target_split=split)
                    
                    # revert the 'observed' variable
                    self.__base_model.classifier.observed.copy_(curr_observed_mask)
            
            # training loop for the current task
            for epoch_cnt in range(epoch_per_task):
                for exp_name in ['exp', 'base', 'accum'] if self.full_mode else ['exp']:
                    if stop_training[exp_name]: continue
                    train_stats = {}
                    
                    # training phase of the current epoch
                    # users do not need to consider collecting and reducing the training stats in the current epoch. Instead, ``BeGin`` automatically collects and reduces them and compute the final training stats for the current epoch.
                    models[exp_name].train()
                    for curr_batch in iter(dataloaders[exp_name]['train']):
                        # handle each minibatch
                        self._trainWrapper(models[exp_name], optims[exp_name], curr_batch, training_states[exp_name], train_stats)
                    # reduce the training stats. The default behavior is averaging the values.
                    reduced_train_stats = self._reduceTrainingStats(train_stats)
                    
                    # evaluation phase of the current epoch
                    models[exp_name].eval()
                    val_stats, val_predictions = {}, []
                    for curr_batch in iter(dataloaders[exp_name]['val']):
                        # handle each minibatch
                        val_predictions.append(self._evalWrapper(models[exp_name], curr_batch, val_stats))
                    # reduce the validation stats. The default behavior is averaging the values.
                    reduced_val_stats = self._reduceEvalStats(val_stats)
                    
                    # compute the current performance on validation set
                    if exp_name == 'accum':
                        val_metric_result = self.__scenario.get_accum_eval_result(torch.cat(val_predictions, dim=0), target_split='val')[-1].item()
                    else:
                        val_metric_result = self.__scenario.get_eval_result(torch.cat(val_predictions, dim=0), target_split='val')[self.__scenario._curr_task].item()
                    
                    # handle procedure for printing training logs. BeGin provides reduced stats obtained from the train and validation dataset
                    if exp_name == 'exp':
                        self.processTrainingLogs(self.__scenario._curr_task, epoch_cnt, val_metric_result, reduced_train_stats, reduced_val_stats)
                    
                    curr_iter_results = {'val_metric': val_metric_result, 'train_stats': reduced_train_stats, 'val_stats': reduced_val_stats}
                    
                    # handle procedure for after each itearation and determine whether to continue training or not
                    if not self.processAfterEachIteration(models[exp_name], optims[exp_name], training_states[exp_name], curr_iter_results):
                        stop_training[exp_name] = True
            
            # handle procedure for right after the training ends
            for exp_name in ['base', 'accum', 'exp'] if self.full_mode else ['exp']:
                models[exp_name].eval()
                self.processAfterTraining(self.__scenario._curr_task, curr_dataset, models[exp_name], optims[exp_name], training_states[exp_name])
                    
            # measure the performance on (accumulated) validation/test dataset
            for split in ['val', 'test']:
                for exp_name in ['base', 'accum', 'exp'] if self.full_mode else ['exp']:
                    models[exp_name].eval()
                    test_predictions, test_stats = [], {}
                    for curr_batch in iter(dataloaders['accum'][split]):
                        # handle each minibatch
                        test_predictions.append(self._evalWrapper(models[exp_name], curr_batch, test_stats))
                        
                    # measure the performance using the collected prediction results
                    test_predictions = torch.cat(test_predictions, dim=0)
                    if exp_name == 'exp' and split == 'test':
                        eval_results = self.__scenario.next_task(test_predictions)
                    elif split == 'val':
                        base_eval_results[exp_name][split].append(self.__scenario.get_accum_eval_result(test_predictions, target_split=split))
                    else:
                        # test dataset is already accumulated
                        base_eval_results[exp_name][split].append(self.__scenario.get_eval_result(test_predictions, target_split=split))
        
        # return the final evaluationr results
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
    
    # event functions and their helper/wrapper functions
    def initTrainingStates(self, dataset, model, optimizer):
        """
            Initialize the dictionary for storing training states (i.e., intermedeiate results).

            Returns:
                Initialized training state (dict).
        """
        return {}
    
    def prepareLoader(self, curr_dataset, curr_training_states):
        """
            Returns:
                Train, valid, and test Dataloaders according to graph problems.
        """
        raise NotImplementedError
        
    def processBeforeTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        """
            Execute some processes before training.

            Note:
                For example, user computes the intermediate statistics for training.

        """
        pass
    
    def beforeInference(self, model, optimizer, _curr_batch, training_states):
        """
            Train a model according to graph problems.
        """
        raise NotImplementedError
    
    def inference(self, model, _curr_batch, training_states):
        """
            Train a model according to graph problems.
        """
        raise NotImplementedError
    
    def afterInference(self, results, model, optimizer, _curr_batch, training_states):
        """
            Train a model according to graph problems.
        """
        raise NotImplementedError
    
    def processTrainIteration(self, model, optimizer, curr_batch, training_states):
        """
            Train a model according to graph problems.
        """
        raise NotImplementedError
    
    def _trainWrapper(self, model, optimizer, curr_batch, training_states, curr_stats):
        """
            The wrapper function for training iteration.
            The main role of the function is to collect the returned dictionary of
            the processTrainItearation function to compute final training stats at every epoch.
        """
        new_stats = self.processTrainIteration(model, optimizer, curr_batch, training_states)
        if new_stats is not None:
            for k, v in new_stats.items():
                if k not in curr_stats:
                    curr_stats[k] = []
                curr_stats[k].append(v)
                
    def _reduceTrainingStats(self, curr_stats):
        """
            The helper function to reduce the returned stats during training.
            The default behavior of the function is to compute average for each key in the returned dictionaries.
        """
        if '_num_items' not in curr_stats:
            reduced_stats = {k: sum(v) / len(v) for k, v in curr_stats.items()}
        else:
            weights = np.array(curr_stats.pop('_num_items'))
            total = weights.sum()
            reduced_stats = {k: (np.array(v) * weights).sum() / total for k, v in curr_stats.items()}
        return reduced_stats
    
    def processEvalIteration(self, model, curr_batch):
        """
            Evaluate a model.
        """
        raise NotImplementedError
    
    def _evalWrapper(self, model, curr_batch, curr_stats):
        """
            The wrapper function for validation/test iteration.
            The main role of the function is to collect the returned dictionary of
            the processEvalItearation function to compute final stats for evalution at every epoch.
        """
        preds, new_stats = self.processEvalIteration(model, curr_batch)
        if new_stats is not None:
            for k, v in new_stats.items():
                if k not in curr_stats:
                    curr_stats[k] = []
                curr_stats[k].append(v)
        return preds
    
    def _reduceEvalStats(self, curr_stats):
        """
            The helper function to reduce the returned stats during evaluation.
            The default behavior of the function is to compute average for each key in the returned dictionaries.
        """
        if '_num_items' not in curr_stats:
            reduced_stats = {k: sum(v) / len(v) for k, v in curr_stats.items()}
        else:
            weights = np.array(curr_stats.pop('_num_items'))
            total = weights.sum()
            reduced_stats = {k: (np.array(v) * weights).sum() / total for k, v in curr_stats.items()}
        return reduced_stats
    
    def processTrainingLogs(self, task_id, epoch_cnt, val_metric_result, train_stats, val_stats):
        """
            Log the intermediate results.
        """
        pass
    
    def processAfterEachIteration(self, curr_model, curr_optimizer, curr_training_states, curr_iter_results):
        """
            Execute some processes after each training iteration.
            Whether to continue training or not is determined by the return value of this function.
            If the returned value is False, the trainer stops training the current model in the current task.
            
            Note:
                This function is called for every end of each epoch,
                and the event function ``processAfterTraining`` is called only when the learning on the current task has ended. 
        """
        return True
    
    def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        """
            Execute some processes after training the current task.

            Note:
                The event function ``processAfterEachIteration`` is called for every end of each epoch,
                and this function is called only when the learning on the current task has ended. 
        """
        pass