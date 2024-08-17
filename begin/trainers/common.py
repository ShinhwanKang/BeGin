import time
import copy
import torch
from torch import nn
import random
import numpy as np
import sys
import dgl
import os

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
        In addition, BaseTrainer supports `benchmark = True` and `seed` (int) to fix the random seed, and `full_mode = True` to additionally evaluate the joint (accum) model. 
    """
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device=None, **kwargs):
        # set random seed
        if kwargs.get('benchmark', False):
            fixed_seed = kwargs.get('seed', 0)
            torch.manual_seed(fixed_seed)
            random.seed(fixed_seed)
            np.random.seed(fixed_seed)
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True
            
        self.__scenario = scenario
        self.__timestamp = str(time.time()).replace('.', '')
        self.__model = model
        self.__optimizer = optimizer_fn(self.__model.parameters())
        self.optimizer_fn = optimizer_fn
        
        # set path for storing initial parameters of model and optimizer
        self.tmp_path = kwargs.get('tmp_save_path', 'tmp')
        self.result_path = kwargs.get('tmp_save_path', 'results')
        
        try:
            os.mkdir(self.tmp_path)
        except:
            pass
        
        try:
            os.mkdir(self.result_path)
        except:
            pass
        
        self.__model_weight_path = f'{self.tmp_path}/init_model_{self.__timestamp}.pkt'
        self.__optim_weight_path = f'{self.tmp_path}/init_optimizer_{self.__timestamp}.pkt'
        self.save_file_name = f'result_{self.__timestamp}'
        
        torch.save(self.__model.state_dict(), self.__model_weight_path)
        torch.save(self.__optimizer.state_dict(), self.__optim_weight_path)
        
        # initial settings for base model and joint model (they are used only when full_mode is enabled)
        self.__base_model = copy.deepcopy(model)
        self.__base_optimizer = optimizer_fn(self.__base_model.parameters())
        self._reset_model(self.__base_model)
        self._reset_optimizer(self.__base_optimizer, self.__base_model)
        self.__accum_model = copy.deepcopy(model)
        self.__accum_optimizer = optimizer_fn(self.__accum_model.parameters())
        self._reset_model(self.__accum_model)
        self._reset_optimizer(self.__accum_optimizer, self.__accum_model)
        
        # other initialization settings 
        self.loss_fn = loss_fn if loss_fn is not None else (lambda x: None) # loss function
        self.device = device # gpu device
        self.num_tasks = scenario.num_tasks # number of tasks
        self.eval_fn = lambda x, y: scenario.get_simple_eval_result(x, y) # evaluation function for minibatches
        self.full_mode = kwargs.get('full_mode', False) # joint model is used when full_mode=True
        self.verbose = kwargs.get('verbose', True)
        self.binary = kwargs.get('binary', False)
        
    @property
    def incr_type(self):
        """ 
            Returns:
                The incremental setting (e.g., task, class, domain, or time).
                The trainer retrieves the value from the given scenario loader.
        """
        return self.__scenario.incr_type
        
    @property
    def curr_task(self):
        """ 
            Returns: 
                The index of a current task (from :math:`0` to :math:`T-1`)
        """
        return self.__scenario._curr_task
    
    def _reset_model(self, target_model):
        """ 
            Reinitialize a model.
            
            Args:
                target_model (torch.nn.Module): a model needed to re-initialize
        """
        target_model.load_state_dict(torch.load(self.__model_weight_path))
        
    def _reset_optimizer(self, target_optimizer, target_model):
        """ 
            Reinitialize an optimizer.
            
            Args:
                target_model (torch.optim.Optimizer): an optimizer needed to re-initialize
        """
        # target_optimizer = self.optimizer_fn(target_model.parameters())
        target_optimizer.load_state_dict(torch.load(self.__optim_weight_path))

    """
    def add_parameters(self, target_optimizer, target_model):
        self._reset_optimizer(target_optimizer, target_model)
        # target_optimizer.add_param_group({'params': target_params})
        # torch.save(target_optimizer.state_dict(), self.__optim_weight_path)
    """
        
    def run(self, epoch_per_task = 1):
        """
            Run the overall process of graph continual learning optimization.
            
            Args:
                epoch_per_task (int): maximum number of training epochs for each task
                
            Returns:
                The base trainer returns the dictionary containing the evaluation results on validation and test dataset.
                And each trainer for specific problem processes the results and outputs the matrix-shaped results for performances 
                and the final evaluation metrics, such as AP, AF, INT, and FWT.
        """
        self.max_num_epochs = epoch_per_task
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
            self._reset_optimizer(self.__base_optimizer, self.__base_model)
            training_states['accum'] = copy.deepcopy(initial_training_state)
            self._reset_model(self.__accum_model)
            self._reset_optimizer(self.__accum_optimizer, self.__accum_model)
            
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
                if exp_name == 'exp' or self.curr_task == 0 or self.full_mode:
                    self.processBeforeTraining(self.__scenario._curr_task, curr_dataset, models[exp_name], optims[exp_name], training_states[exp_name])
            dataloaders['accum'] = {k: v for k, v in zip(['train', 'val', 'test'], self.prepareLoader(accumulated_dataset, training_states['accum']))}
            if self.full_mode:
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
                        for curr_batch in dataloaders['base'][split]:
                            initial_test_predictions.append(self._evalWrapper(models['base'], curr_batch, initial_stats))
                        # compute the initial performances
                        initial_results[split] = self.__scenario.get_eval_result(torch.cat(initial_test_predictions, dim=0), target_split=split)
                    
                    # revert the 'observed' variable
                    self.__base_model.classifier.observed.copy_(curr_observed_mask)
                    
                    # we need to send initial result on the test dataset to compute FWT in the scenario loader
                    if self.incr_type == 'domain':
                        self.__scenario.initial_test_result = initial_results['test']
                    
            # training loop for the current task
            for epoch_cnt in range(epoch_per_task):
                for exp_name in ['exp', 'base', 'accum'] if self.full_mode else ['exp']:
                    if stop_training[exp_name]: continue
                    train_stats = {}
                    
                    # training phase of the current epoch
                    # users do not need to consider collecting and reducing the training stats in the current epoch. Instead, ``BeGin`` automatically collects and reduces them and compute the final training stats for the current epoch.
                    models[exp_name].train()
                    for curr_batch in dataloaders[exp_name]['train']:
                        # handle each minibatch
                        self._trainWrapper(models[exp_name], optims[exp_name], curr_batch, training_states[exp_name], train_stats)
                    # reduce the training stats. The default behavior is averaging the values.
                    reduced_train_stats = self._reduceTrainingStats(train_stats)
                    
                    # evaluation phase of the current epoch
                    models[exp_name].eval()
                    val_stats, val_predictions = {}, []
                    for curr_batch in dataloaders[exp_name]['val']:
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
                    if exp_name == 'exp' and self.verbose:
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
                    for curr_batch in dataloaders['accum'][split]:
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
        
        # return the final evaluation results
        if self.full_mode:
            return {'init_val': initial_results['val'],
                    'init_test': initial_results['test'],
                    'exp_val': torch.stack(base_eval_results['exp']['val'], dim=0),
                    'exp_test': eval_results['exp_results'],
                    'exp_AP': eval_results['AP'],
                    'exp_AF': eval_results['AF'],
                    'exp_FWT': eval_results['FWT'],
                    'base_val': torch.stack(base_eval_results['base']['val'], dim=0),
                    'base_test': torch.stack(base_eval_results['base']['test'], dim=0),
                    'accum_val': torch.stack(base_eval_results['accum']['val'], dim=0),
                    'accum_test': torch.stack(base_eval_results['accum']['test'], dim=0),
                   }
        else:
            return {'init_val': initial_results['val'],
                    'init_test': initial_results['test'],
                    'exp_val': torch.stack(base_eval_results['exp']['val'], dim=0),
                    'exp_test': eval_results['exp_results'],
                    'exp_AP': eval_results['AP'],
                    'exp_AF': eval_results['AF'],
                    'exp_FWT': eval_results['FWT'],
                   }
    
    # event functions and their helper/wrapper functions
    def initTrainingStates(self, scenario, model, optimizer):
        """
            The event function to initialize the dictionary for storing training states (i.e., intermedeiate results).
            
            Args:
                scenario (begin.scenarios.common.BaseScenarioLoader): the given ScenarioLoader to the trainer
                model (torch.nn.Module): the given model to the trainer
                optmizer (torch.optim.Optimizer): the optimizer generated from the given `optimizer_fn` 
                
            Returns:
                Initialized training state (dict).
        """
        return {}
    
    def prepareLoader(self, curr_dataset, curr_training_states):
        """
            The event function to generate dataloaders from the given dataset for the current task.
            
            Args:
                curr_dataset (object): The dataset for the current task. Its type is dgl.graph for node-level and link-level problem, and dgl.data.DGLDataset for graph-level problem.
                curr_training_states (dict): the dictionary containing the current training states.
                
            Returns:
                A tuple containing three dataloaders.
                The trainer considers the first dataloader, second dataloader, and third dataloader
                as dataloaders for training, validation, and test, respectively.
        """
        raise NotImplementedError
        
    def processBeforeTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        """
            The event function to execute some processes before training.
            
            Args:
                task_id (int): the index of the current task
                curr_dataset (object): The dataset for the current task.
                curr_model (torch.nn.Module): the current trained model.
                curr_optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_training_states (dict): the dictionary containing the current training states.
        """
        pass
    
    def predictionFormat(self, results):
        """
            The helper function for formatting the prediction results before feeding the results to evaluator.
            
            Args:
                results (dict): the dictionary containing the prediction results.
        """
        pass
        
    def beforeInference(self, model, optimizer, curr_batch, curr_training_states):
        """
            The event function to execute some processes right before inference (for training).
            
            Args:
                model (torch.nn.Module): the current trained model.
                optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
        """
        raise NotImplementedError
    
    def inference(self, model, curr_batch, curr_training_states):
        """
            The event function to execute inference step.
            
            Args:
                model (torch.nn.Module): the current trained model.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
                
            Returns:
                A dictionary containing the inference results, such as prediction result and loss.
        """
        raise NotImplementedError
    
    def afterInference(self, results, model, optimizer, curr_batch, curr_training_states):
        """
            The event function to execute some processes right after the inference step (for training).
            We recommend performing backpropagation in this event function.
            
            Args:
                results (dict): the returned dictionary from the event function `inference`.
                model (torch.nn.Module): the current trained model.
                optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
                
            Returns:
                A dictionary containing the information from the `results`.
        """
        raise NotImplementedError
    
    def processTrainIteration(self, model, optimizer, curr_batch, curr_training_states):
        """
            The event function to handle every training iteration.
            
            Args:
                model (torch.nn.Module): the current trained model.
                optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
                
            Returns:
                A dictionary containing the outcomes (stats) during the training iteration.
        """
        raise NotImplementedError
    
    def _trainWrapper(self, model, optimizer, curr_batch, curr_training_states, curr_stats):
        """
            The wrapper function for training iteration.
            The main role of the function is to collect the returned dictionary of
            the processTrainItearation function to compute final training stats at every epoch.
            
            Args:
                model (torch.nn.Module): the current trained model.
                optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
                curr_stats (dict): the dictionary to store the returned dictionaries.
        """
        new_stats = self.processTrainIteration(model, optimizer, curr_batch, curr_training_states)
        if new_stats is not None:
            for k, v in new_stats.items():
                if k not in curr_stats:
                    curr_stats[k] = []
                curr_stats[k].append(v)
                
    def _reduceTrainingStats(self, curr_stats):
        """
            The helper function to reduce the returned stats during training.
            The default behavior of the function is to compute average for each key in the returned dictionaries.
            
            Args:
                curr_stats (dict): the dictionary containing the training stats.
            
            Returns:
                A reduced dictionary containing the final training outcomes.
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
            The event function to handle every evaluation iteration.
            
            Args:
                model (torch.nn.Module): the current trained model.
                curr_batch (object): the data (or minibatch) for the current iteration.
                
            Returns:
                A dictionary containing the outcomes (stats) during the evaluation iteration.
        """
        raise NotImplementedError
    
    def _evalWrapper(self, model, curr_batch, curr_stats):
        """
            The wrapper function for validation/test iteration.
            The main role of the function is to collect the returned dictionary of
            the processEvalItearation function to compute final stats for evalution at every epoch.
            
            Args:
                model (torch.nn.Module): the current trained model.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_stats (dict): the dictionary to store the returned dictionaries.
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
            
            Args:
                curr_stats (dict): the dictionary containing the evaluation stats.
            
            Returns:
                A reduced dictionary containing the final evaluation outcomes.
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
            (Optional) The event function to output the training logs.
            
            Args:
                task_id (int): the index of the current task
                epoch_cnt (int): the index of the current epoch
                val_metric_result (object): the validation performance computed by the evaluator
                train_stats (dict): the reduced dictionary containg the final training outcomes.
                val_stats (dict): the reduced dictionary containg the final validation outcomes.
        """
        pass
    
    def processAfterEachIteration(self, curr_model, curr_optimizer, curr_training_states, curr_iter_results):
        """
            The event function to execute some processes for every end of each epoch.
            Whether to continue training or not is determined by the return value of this function.
            If the returned value is False, the trainer stops training the current model in the current task.
            
            Note:
                This function is called for every end of each epoch,
                and the event function ``processAfterTraining`` is called only when the learning on the current task has ended. 
                
            Args:
                curr_model (torch.nn.Module): the current trained model.
                curr_optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_training_states (dict): the dictionary containing the current training states.
                curr_iter_results (dict): the dictionary containing the training/validation results of the current epoch.
                
            Returns:
                A boolean value. If the returned value is False, the trainer stops training the current model in the current task.
        """
        return True
    
    def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        """
            The event function to execute some processes after training the current task.

            Note:
                The event function ``processAfterEachIteration`` is called for every end of each epoch,
                and this function is called only when the learning on the current task has ended.
                
            Args:
                task_id (int): the index of the current task.
                curr_dataset (object): The dataset for the current task.
                curr_model (torch.nn.Module): the current trained model.
                curr_optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_training_states (dict): the dictionary containing the current training states.
        """
        pass