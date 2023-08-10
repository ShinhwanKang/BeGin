import torch
import copy
import torch.nn.functional as F
from begin.trainers.nodes import NCTrainer, NCMinibatchTrainer
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
                'acc': self.eval_fn(self.predictionFormat(results), _curr_batch[0].ndata['label'][_curr_batch[1]].to(self.device))}
    
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
                'acc': self.eval_fn(self.predictionFormat(results), _curr_batch[0].ndata['label'][_curr_batch[1]].to(self.device))}
        
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

class NCClassILGEMMinibatchTrainer(NCMinibatchTrainer):
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
            for train_loader in training_states['memories']:
                model.zero_grad()
                all_grads.append(None)
                for mem_batch in train_loader:
                    pre_results = self.inference(model, mem_batch, training_states)
                    pre_results['loss'].backward()
                    all_grads[-1] = torch.cat([p.grad.data.clone().view(-1) for p in model.parameters()])
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
                'acc': self.eval_fn(self.predictionFormat(results), _curr_batch[-1][-1].dstdata['label'].to(self.device))}
        
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
        candidates = torch.nonzero(curr_dataset.ndata['train_mask'], as_tuple=True)[0]
        perm = torch.randperm(candidates.shape[0])
        
        g_train = torch.Generator()
        g_train.manual_seed(0)
        train_sampler = dgl.dataloading.MultiLayerNeighborSampler([5, 10, 10])
        train_loader = dgl.dataloading.NodeDataLoader(
            curr_dataset, candidates[perm[:self.num_memories]], train_sampler,
            batch_size=131072,
            shuffle=True,
            drop_last=False,
            num_workers=1, worker_init_fn=self._dataloader_seed_worker, generator=g_train)
        
        curr_training_states['memories'].append(train_loader)
        
class NCDomainILGEMTrainer(NCClassILGEMTrainer):    
    """
        This trainer has the same behavior as `NCClassILGEMTrainer`.
    """
    pass

class NCTimeILGEMTrainer(NCClassILGEMTrainer):
    """
        This trainer has the same behavior as `NCClassILGEMTrainer`.
    """
    pass
