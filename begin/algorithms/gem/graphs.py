import torch
import dgl
import copy
from itertools import chain
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
                'acc': self.eval_fn(self.predictionFormat(results), labels.to(self.device))}
    
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
                'acc': self.eval_fn(self.predictionFormat(results), labels.to(self.device))}
    
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

class GCDomainILGEMTrainer(GCClassILGEMTrainer):
    """
        This trainer has the same behavior as `GCClassILGEMTrainer`.
    """
    pass
        
class GCTimeILGEMTrainer(GCClassILGEMTrainer):
    """
        This trainer has the same behavior as `GCClassILGEMTrainer`.
    """
    pass