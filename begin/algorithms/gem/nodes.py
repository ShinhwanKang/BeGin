import sys
import numpy as np
import torch
import copy
import torch.nn.functional as F
from begin.trainers.nodes import NCTrainer
from .utils import project2cone2

class NCTaskILGEMTrainer(NCTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        """
            GEM needs `lamb` and `num_memories`, the additional hyperparamters for quadratic programming and the training buffer size, respectively.
        """
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.lamb = kwargs['lamb'] if 'lamb' in kwargs else .5
        self.num_memories = kwargs['num_memories'] if 'num_memories' in kwargs else 100
        self.num_memories = (self.num_memories // self.num_tasks)
        
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
        curr_batch, mask = _curr_batch
        preds = model(curr_batch.to(self.device), curr_batch.ndata['feat'].to(self.device), task_masks=curr_batch.ndata['task_specific_mask'].to(self.device))[mask]
        loss = self.loss_fn(preds, curr_batch.ndata['label'][mask].to(self.device))
        return {'preds': preds, 'loss': loss}
    
    def initTrainingStates(self, scenario, model, optimizer):
        return {'memories': []}
    
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
            for mem in training_states['memories']:
                model.zero_grad()
                mem_mask = torch.zeros_like(_curr_batch[1])
                mem_mask[mem['nodes']] = True
                mem_task_mask = torch.zeros_like(_curr_batch[0].ndata['task_specific_mask'])
                mem_task_mask[mem['nodes']] = mem['task_specific_mask']
                mem_batch = (copy.deepcopy(_curr_batch[0]), mem_mask)
                mem_batch[0].ndata['task_specific_mask'] = mem_task_mask
                
                pre_results = self.inference(model, mem_batch, training_states)
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
        return {'loss': results['loss'].item(),
                'acc': self.eval_fn(results['preds'].argmax(-1), _curr_batch[0].ndata['label'][_curr_batch[1]].to(self.device))}
    
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
        train_loader = self.prepareLoader(curr_dataset, curr_training_states)[0]
        chosen_nodes = []
        chosen_nodes_mask = []
        for i, _curr_batch in enumerate(iter(train_loader)):
            curr_batch, mask = _curr_batch
            candidates = torch.nonzero(mask, as_tuple=True)[0]
            perm = torch.randperm(candidates.shape[0])
            new_chosen_nodes = candidates[perm[:self.num_memories]]
            chosen_nodes.append(new_chosen_nodes)
            chosen_nodes_mask.append(curr_batch.ndata['task_specific_mask'][new_chosen_nodes])
        curr_training_states['memories'].append({'nodes': torch.cat(chosen_nodes, dim=-1), 'task_specific_mask': torch.cat(chosen_nodes_mask, dim=0)})

class NCClassILGEMTrainer(NCTrainer):
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
            for mem in training_states['memories']:
                model.zero_grad()
                mem_mask = torch.zeros_like(_curr_batch[1])
                mem_mask[mem] = True
                mem_batch = (copy.deepcopy(_curr_batch[0]), mem_mask)
                pre_results = self.inference(model, mem_batch, training_states)
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
        return {'loss': results['loss'].item(),
                'acc': self.eval_fn(results['preds'].argmax(-1), _curr_batch[0].ndata['label'][_curr_batch[1]].to(self.device))}
        
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
        train_loader = self.prepareLoader(curr_dataset, curr_training_states)[0]
        chosen_nodes = []
        for i, _curr_batch in enumerate(iter(train_loader)):
            curr_batch, mask = _curr_batch
            candidates = torch.nonzero(mask, as_tuple=True)[0]
            perm = torch.randperm(candidates.shape[0])
            chosen_nodes.append(candidates[perm[:self.num_memories]])
        curr_training_states['memories'].append(torch.cat(chosen_nodes, dim=-1))
        
class NCDomainILGEMTrainer(NCTrainer):
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
        curr_batch, mask = _curr_batch
        
        if len(training_states['memories']) > 0:
            all_grads = []
            for mem in training_states['memories']:
                model.zero_grad()
                mem_mask = torch.zeros_like(mask)
                mem_mask[mem] = True
                preds = model(curr_batch.to(self.device), curr_batch.ndata['feat'].to(self.device))[mem_mask]
                loss = self.loss_fn(preds, curr_batch.ndata['label'][mem_mask].to(self.device))
                loss.backward()
                all_grads.append(torch.cat([p.grad.data.clone().view(-1) for p in model.parameters()]))
            all_grads = torch.stack(all_grads, dim=0)
                
        optimizer.zero_grad()
        preds = model(curr_batch.to(self.device), curr_batch.ndata['feat'].to(self.device))[mask]
        loss = self.loss_fn(preds, curr_batch.ndata['label'][mask].to(self.device))
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
        return {'loss': loss.item(), 'acc': self.eval_fn(preds, curr_batch.ndata['label'][mask].to(self.device))}
    
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
        curr_batch, mask = _curr_batch
        preds = model(curr_batch.to(self.device), curr_batch.ndata['feat'].to(self.device))[mask]
        loss = self.loss_fn(preds, curr_batch.ndata['label'][mask].float().to(self.device))
        return preds, {'loss': loss.item()}
    
    def processBeforeTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        """
            The event function to execute some processes before training.
            
            We need to extend the base function since the output format is slightly different from the base trainer.
        
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
        curr_model.observe_labels(torch.arange(curr_dataset.ndata['label'].shape[-1]))
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
        train_loader = self.prepareLoader(curr_dataset, curr_training_states)[0]
        chosen_nodes = []
        for i, _curr_batch in enumerate(iter(train_loader)):
            curr_batch, mask = _curr_batch
            candidates = torch.nonzero(mask, as_tuple=True)[0]
            perm = torch.randperm(candidates.shape[0])
            chosen_nodes.append(candidates[perm[:self.num_memories]])
        curr_training_states['memories'].append(torch.cat(chosen_nodes, dim=-1))
        
class NCTimeILGEMTrainer(NCClassILGEMTrainer):
    pass
