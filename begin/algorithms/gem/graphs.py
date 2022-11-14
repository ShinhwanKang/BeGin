import sys
import random
import numpy as np
import torch
import copy
import torch.nn.functional as F

from begin.trainers.graphs import GCTrainer
from .utils import project2cone2

class GCTaskILGEMTrainer(GCTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        """
            GEM needs `lamb` and `num_memories`, the additional hyperparamters for quadratic programming and the training buffer size, respectively.
        """
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.lamb = kwargs['lamb'] if 'lamb' in kwargs else .5
        self.num_memories = kwargs['num_memories'] if 'num_memories' in kwargs else 100
        self.num_memories = (self.num_memories // self.num_tasks)
        
    def initTrainingStates(self, scenario, model, optimizer):
        return {'memories': []}
    
    def processBeforeTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        """
            The event function to execute some processes before training.
            
            GEM performs initialization (for every task) to manage the memory
        
            Args:
                task_id (int): the index of the current task
                curr_dataset (object): The dataset for the current task.
                curr_model (torch.nn.Module): the current trained model.
                curr_optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_training_states (dict): the dictionary containing the current training states.
        """
        super().processBeforeTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)
        curr_training_states['curr_memory'] = []
        curr_training_states['curr_memory_size'] = 0
        
    def inference(self, model, _curr_batch, training_states):
        """
            The event function to execute inference step.
        
            For task-IL, we need to additionally consider task information for the inference step.
        
            Args:
                model (torch.nn.Module): the current trained model.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
                
            Returns:
                A dictionary containing the inference results, such as prediction result and loss.
        """
        graphs, labels, masks = _curr_batch
        preds = model(graphs.to(self.device),
                      graphs.ndata['feat'].to(self.device) if 'feat' in graphs.ndata else None,
                      edge_attr = graphs.edata['feat'].to(self.device) if 'feat' in graphs.edata else None,
                      edge_weight = graphs.edata['weight'].to(self.device) if 'weight' in graphs.edata else None,
                      task_masks = masks)
        loss = self.loss_fn(preds, labels.to(self.device))
        return {'preds': preds, 'loss': loss}
    
    def beforeInference(self, model, optimizer, _curr_batch, training_states):
        """
            The event function to execute some processes right before inference (for training).
            
            GEM computes the gradients for the previous tasks using the sampled data stored in the memory.
            
            Args:
                model (torch.nn.Module): the current trained model.
                optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
        """
        if len(training_states['memories']) > 0:
            all_grads = []
            for saved_batch in training_states['memories']:
                model.zero_grad()
                pre_results = self.inference(model, saved_batch, training_states)
                pre_results['loss'].backward()
                all_grads.append(torch.cat([p.grad.data.clone().view(-1) for p in model.parameters()]))
            training_states['all_grads'] = torch.stack(all_grads, dim=0)
        model.zero_grad()
    
    def afterInference(self, results, model, optimizer, _curr_batch, training_states):
        """
            The event function to execute some processes right after the inference step (for training).
            We recommend performing backpropagation in this event function.
        
            Using the computed gradients from the samples, GEM controls the gradients for the current task with quadratic programming.
        
            Args:
                results (dict): the returned dictionary from the event function `inference`.
                model (torch.nn.Module): the current trained model.
                optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
                
            Returns:
                A dictionary containing the information from the `results`.
        """
        results['loss'].backward()
        if len(training_states['memories']) > 0:
            curr_grad = torch.cat([p.grad.data.view(-1) for p in model.parameters()])
            if ((training_states['all_grads'] * curr_grad).sum(-1) < 0).any():
                new_gradient = project2cone2(curr_grad, training_states['all_grads'], self.lamb)
                curr_idx = 0
                for p in model.parameters():
                    p_size = p.data.numel()
                    p.grad.copy_(new_gradient[curr_idx:(curr_idx + p_size)].view_as(p.data))
                    curr_idx += p_size            
        optimizer.step()
        
        graphs, labels, masks = _curr_batch
        training_states['curr_memory'].append({'graphs': graphs, 'labels': labels, 'tmasks': masks, 'ranges': (0, graphs.batch_size)})
        training_states['curr_memory_size'] += graphs.batch_size
        while training_states['curr_memory_size'] > self.num_memories:
            _from, _to = training_states['curr_memory'][0]['ranges']
            _diff = training_states['curr_memory_size'] - self.num_memories
            if _diff >= (_to - _from):
                training_states['curr_memory'] = training_states['curr_memory'][1:]
                training_states['curr_memory_size'] -= (_to - _from)
            else:
                training_states['curr_memory'][0]['ranges'] = (_from + _diff, _to)
                training_states['curr_memory_size'] -= _diff
                break       
        return {'_num_items': results['preds'].shape[0],
                'loss': results['loss'].item(),
                'acc': self.eval_fn(results['preds'].argmax(-1), labels.to(self.device))}
    
    def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        """
            The event function to execute some processes after training the current task.

            GEM samples the instances in the training dataset for computing gradients in :func:`beforeInference` (or :func:`processTrainIteration`) for the future tasks.
                
            Args:
                task_id (int): the index of the current task.
                curr_dataset (object): The dataset for the current task.
                curr_model (torch.nn.Module): the current trained model.
                curr_optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_training_states (dict): the dictionary containing the current training states.
        """
        super().processAfterTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)
        
        chosen_graphs = list(chain.from_iterable([dgl.unbatch(mem['graphs']) for mem in curr_training_states['curr_memory']]))
        chosen_labels = torch.cat([mem['labels'] for mem in curr_training_states['curr_memory']], dim=0)
        chosen_masks = torch.cat([mem['tmasks'] for mem in curr_training_states['curr_memory']], dim=0)
        
        _from = curr_training_states['curr_memory'][0]['ranges'][0]
        chosen_graphs = dgl.batch(chosen_graphs[_from:])
        chosen_labels = chosen_labels[_from:]
        chosen_masks = chosen_masks[_from:]
        
        print(chosen_graphs.batch_size, chosen_labels.shape, chosen_masks.shape)
        curr_training_states['memories'].append((chosen_graphs, chosen_labels, chosen_masks))
        
class GCClassILGEMTrainer(GCTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        """
            GEM needs `lamb` and `num_memories`, the additional hyperparamters for quadratic programming and the training buffer size, respectively.
        """
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.lamb = kwargs['lamb'] if 'lamb' in kwargs else .5
        self.num_memories = kwargs['num_memories'] if 'num_memories' in kwargs else 100
        self.num_memories = (self.num_memories // self.num_tasks)
        
    def initTrainingStates(self, scenario, model, optimizer):
        return {'memories': []}
    
    def processBeforeTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        """
            The event function to execute some processes before training.
            
            GEM performs initialization (for every task) to manage the memory
        
            Args:
                task_id (int): the index of the current task
                curr_dataset (object): The dataset for the current task.
                curr_model (torch.nn.Module): the current trained model.
                curr_optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_training_states (dict): the dictionary containing the current training states.
        """
        super().processBeforeTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)
        curr_training_states['curr_memory'] = []
        curr_training_states['curr_memory_size'] = 0
        
    def beforeInference(self, model, optimizer, _curr_batch, training_states):
        """
            The event function to execute some processes right before inference (for training).
            
            GEM computes the gradients for the previous tasks using the sampled data stored in the memory.
            
            Args:
                model (torch.nn.Module): the current trained model.
                optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
        """
        if len(training_states['memories']) > 0:
            all_grads = []
            for saved_batch in training_states['memories']:
                model.zero_grad()
                pre_results = self.inference(model, saved_batch, training_states)
                pre_results['loss'].backward()
                all_grads.append(torch.cat([p.grad.data.clone().view(-1) for p in model.parameters()]))
            training_states['all_grads'] = torch.stack(all_grads, dim=0)
        model.zero_grad()
    
    def afterInference(self, results, model, optimizer, _curr_batch, training_states):
        """
            The event function to execute some processes right after the inference step (for training).
            We recommend performing backpropagation in this event function.
        
            Using the computed gradients from the samples, GEM controls the gradients for the current task with quadratic programming.
        
            Args:
                results (dict): the returned dictionary from the event function `inference`.
                model (torch.nn.Module): the current trained model.
                optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
                
            Returns:
                A dictionary containing the information from the `results`.
        """
        results['loss'].backward()
        if len(training_states['memories']) > 0:
            curr_grad = torch.cat([p.grad.data.view(-1) for p in model.parameters()])
            if ((training_states['all_grads'] * curr_grad).sum(-1) < 0).any():
                new_gradient = project2cone2(curr_grad, training_states['all_grads'], self.lamb)
                curr_idx = 0
                for p in model.parameters():
                    p_size = p.data.numel()
                    p.grad.copy_(new_gradient[curr_idx:(curr_idx + p_size)].view_as(p.data))
                    curr_idx += p_size            
        optimizer.step()
        
        graphs, labels = _curr_batch
        training_states['curr_memory'].append({'graphs': graphs, 'labels': labels, 'ranges': (0, graphs.batch_size)})
        training_states['curr_memory_size'] += graphs.batch_size
        while training_states['curr_memory_size'] > self.num_memories:
            _from, _to = training_states['curr_memory'][0]['ranges']
            _diff = training_states['curr_memory_size'] - self.num_memories
            if _diff >= (_to - _from):
                training_states['curr_memory'] = training_states['curr_memory'][1:]
                training_states['curr_memory_size'] -= (_to - _from)
            else:
                training_states['curr_memory'][0]['ranges'] = (_from + _diff, _to)
                training_states['curr_memory_size'] -= _diff
                break       
        return {'_num_items': results['preds'].shape[0],
                'loss': results['loss'].item(),
                'acc': self.eval_fn(results['preds'].argmax(-1), labels.to(self.device))}
    
    def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        """
            The event function to execute some processes after training the current task.

            GEM samples the instances in the training dataset for computing gradients in :func:`beforeInference` (or :func:`processTrainIteration`) for the future tasks.
                
            Args:
                task_id (int): the index of the current task.
                curr_dataset (object): The dataset for the current task.
                curr_model (torch.nn.Module): the current trained model.
                curr_optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_training_states (dict): the dictionary containing the current training states.
        """
        super().processAfterTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)
        
        chosen_graphs = list(chain.from_iterable([dgl.unbatch(mem['graphs']) for mem in curr_training_states['curr_memory']]))
        chosen_labels = torch.cat([mem['labels'] for mem in curr_training_states['curr_memory']], dim=0)
        
        _from = curr_training_states['curr_memory'][0]['ranges'][0]
        chosen_graphs = dgl.batch(chosen_graphs[_from:])
        chosen_labels = chosen_labels[_from:]
        curr_training_states['memories'].append((chosen_graphs, chosen_labels))

class GCDomainILGEMTrainer(GCTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        """
            GEM needs `lamb` and `num_memories`, the additional hyperparamters for quadratic programming and the training buffer size, respectively.
        """
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.lamb = kwargs['lamb'] if 'lamb' in kwargs else 1.
        self.num_memories = kwargs['num_memories'] if 'num_memories' in kwargs else 100
        self.num_memories = (self.num_memories // self.num_tasks)
        
    def processTrainIteration(self, model, optimizer, _curr_batch, training_states):
        """
            The event function to handle every training iteration.
        
            GEM computes the gradients for the previous tasks using the sampled data stored in the memory.
            Using the computed gradients from the samples, GEM controls the gradients for the current task with quadratic programming.
        
            Args:
                model (torch.nn.Module): the current trained model.
                optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
                
            Returns:
                A dictionary containing the outcomes (stats) during the training iteration.
        """
        if len(training_states['memories']) > 0:
            all_grads = []
            for saved_batch in training_states['memories']:
                _saved_graphs, _saved_labels = saved_batch
                model.zero_grad()
                preds = model(_saved_graphs.to(self.device),
                              _saved_graphs.ndata['feat'].to(self.device) if 'feat' in _saved_graphs.ndata else None,
                              edge_attr = _saved_graphs.edata['feat'].to(self.device) if 'feat' in _saved_graphs.edata else None,
                              edge_weight = _saved_graphs.edata['weight'].to(self.device) if 'weight' in _saved_graphs.edata else None)
                loss = self.loss_fn(preds, _saved_labels.to(self.device))
                loss.backward()
                all_grads.append(torch.cat([p.grad.data.clone().view(-1) for p in model.parameters()]))
            all_grads = torch.stack(all_grads, dim=0)
        
        graphs, labels = _curr_batch
        optimizer.zero_grad()
        preds = model(graphs.to(self.device),
                      graphs.ndata['feat'].to(self.device) if 'feat' in graphs.ndata else None,
                      edge_attr = graphs.edata['feat'].to(self.device) if 'feat' in graphs.edata else None,
                      edge_weight = graphs.edata['weight'].to(self.device) if 'weight' in graphs.edata else None)
        loss = self.loss_fn(preds, labels.to(self.device))
        loss.backward()
        
        if len(training_states['memories']) > 0:
            curr_grad = torch.cat([p.grad.data.view(-1) for p in model.parameters()])
            if ((all_grads * curr_grad).sum(-1) < 0).any():
                new_gradient = project2cone2(curr_grad, all_grads, self.lamb)
                curr_idx = 0
                for p in model.parameters():
                    p_size = p.data.numel()
                    p.grad.copy_(new_gradient[curr_idx:(curr_idx + p_size)].view_as(p.data))
                    curr_idx += p_size            
        optimizer.step()
        
        training_states['curr_memory']['graphs'].append(graphs)
        training_states['curr_memory']['labels'].append(labels)
        training_states['curr_memory']['ranges'].append((0, graphs.batch_size))
        training_states['curr_memory']['size'] += graphs.batch_size
        while training_states['curr_memory']['size'] > self.num_memories:
            _from, _to = training_states['curr_memory']['ranges'][0]
            _diff = training_states['curr_memory']['size'] - self.num_memories
            if _diff >= (_to - _from):
                training_states['curr_memory']['graphs'] = training_states['curr_memory']['graphs'][1:] 
                training_states['curr_memory']['labels'] = training_states['curr_memory']['labels'][1:]
                training_states['curr_memory']['ranges'] = training_states['curr_memory']['ranges'][1:]
                training_states['curr_memory']['size'] -= (_to - _from)
            else:
                training_states['curr_memory']['ranges'][0] = (_from + _diff, _to)
                training_states['curr_memory']['size'] -= _diff
                break
        
        return {'_num_items': preds.shape[0], 'loss': loss.item(), 'acc': self.eval_fn(preds, labels.to(self.device))}
        
    def processEvalIteration(self, model, _curr_batch):
        """
            The event function to handle every evaluation iteration.
            
            We need to extend the function since the output format is slightly different from the base trainer.
        
            Args:
                model (torch.nn.Module): the current trained model.
                curr_batch (object): the data (or minibatch) for the current iteration.
                
            Returns:
                A dictionary containing the outcomes (stats) during the evaluation iteration.
        """
        graphs, labels = _curr_batch
        preds = model(graphs.to(self.device),
                      graphs.ndata['feat'].to(self.device) if 'feat' in graphs.ndata else None,
                      edge_attr = graphs.edata['feat'].to(self.device) if 'feat' in graphs.edata else None,
                      edge_weight = graphs.edata['weight'].to(self.device) if 'weight' in graphs.edata else None)
        loss = self.loss_fn(preds, labels.to(self.device))
        return preds, {'_num_items': preds.shape[0], 'loss': loss.item(), 'acc': self.eval_fn(preds, labels.to(self.device))}
        
    def initTrainingStates(self, scenario, model, optimizer):
        return {'memories': []}
    
    def processBeforeTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        """
            The event function to execute some processes before training.
            
            GEM performs initialization (for every task) to manage the memory
        
            Args:
                task_id (int): the index of the current task
                curr_dataset (object): The dataset for the current task.
                curr_model (torch.nn.Module): the current trained model.
                curr_optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_training_states (dict): the dictionary containing the current training states.
        """
        curr_training_states['scheduler'] = self.scheduler_fn(curr_optimizer)
        curr_training_states['best_val_acc'] = -1.
        curr_training_states['best_val_loss'] = 1e10
        curr_training_states['curr_memory'] = {'graphs': [], 'labels': [], 'ranges': [], 'size': 0}
        curr_model.observe_labels(torch.LongTensor([0]))
        self._reset_optimizer(curr_optimizer)
    
    def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        """
            The event function to execute some processes after training the current task.

            GEM samples the instances in the training dataset for computing gradients in :func:`beforeInference` (or :func:`processTrainIteration`) for the future tasks.
                
            Args:
                task_id (int): the index of the current task.
                curr_dataset (object): The dataset for the current task.
                curr_model (torch.nn.Module): the current trained model.
                curr_optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_training_states (dict): the dictionary containing the current training states.
        """
        super().processAfterTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)
        mem = curr_training_states['curr_memory']
        chosen_graphs = dgl.batch(list(chain.from_iterable([(dgl.unbatch(graphs))[_from:_to] for graphs, (_from, _to) in zip(mem['graphs'], mem['ranges'])])))
        chosen_labels = torch.cat([labels[_from:_to] for labels, (_from, _to) in zip(mem['labels'], mem['ranges'])], dim=0)
        print(chosen_graphs.batch_size, chosen_labels.shape)
        curr_training_states['memories'].append((chosen_graphs, chosen_labels))
        
class GCTimeILGEMTrainer(GCClassILGEMTrainer):
    pass