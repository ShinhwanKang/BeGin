import sys
import numpy as np
import torch
import copy, dgl
import torch.nn.functional as F
from begin.trainers.links import LCTrainer, LPTrainer

class LCTaskILEWCTrainer(LCTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        """
            EWC needs `lamb`, the additional hyperparamter for the regularization term used in :func:`afterInference`
        """
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.lamb = kwargs['lamb'] if 'lamb' in kwargs else 10000.
        
    def prepareLoader(self, _curr_dataset, curr_training_states):
        """
            The event function to generate dataloaders from the given dataset for the current task.
            
            For task-IL, we need to additionally consider task information for the inference step.
            
            Args:
                curr_dataset (object): The dataset for the current task. Its type is dgl.graph for node-level and link-level problem, and dgl.data.DGLDataset for graph-level problem.
                curr_training_states (dict): the dictionary containing the current training states.
                
            Returns:
                A tuple containing three dataloaders.
                The trainer considers the first dataloader, second dataloader, and third dataloader
                as dataloaders for training, validation, and test, respectively.
        """
        curr_dataset = copy.deepcopy(_curr_dataset)
        srcs, dsts = curr_dataset.edges()
        labels = curr_dataset.edata.pop('label')
        train_mask = curr_dataset.edata.pop('train_mask')
        val_mask = curr_dataset.edata.pop('val_mask')
        test_mask = curr_dataset.edata.pop('test_mask')
        task_mask = curr_dataset.edata.pop('task_specific_mask')
        curr_dataset = dgl.add_self_loop(curr_dataset)
        return [(curr_dataset, srcs[train_mask], dsts[train_mask], task_mask[train_mask], labels[train_mask])], [(curr_dataset, srcs[val_mask], dsts[val_mask], task_mask[val_mask], labels[val_mask])], [(curr_dataset, srcs[test_mask], dsts[test_mask], task_mask[test_mask], labels[test_mask])]
    
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
        curr_batch, srcs, dsts, task_masks, labels = _curr_batch
        preds = model(curr_batch.to(self.device), curr_batch.ndata['feat'].to(self.device), srcs, dsts, task_masks=task_masks)
        loss = self.loss_fn(preds, labels.to(self.device))
        return {'preds': preds, 'loss': loss}
    
    def afterInference(self, results, model, optimizer, _curr_batch, training_states):
        """
            The event function to execute some processes right after the inference step (for training).
            We recommend performing backpropagation in this event function.
            
            EWC performs regularization process in this function.
            
            Args:
                results (dict): the returned dictionary from the event function `inference`.
                model (torch.nn.Module): the current trained model.
                optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
                
            Returns:
                A dictionary containing the information from the `results`.
        """
        loss_reg = 0
        for _param, _fisher in zip(training_states['params'], training_states['fishers']):
            for name, p in model.named_parameters():
                l = self.lamb * _fisher[name]
                l = l * ((p - _param[name]) ** 2)
                loss_reg = loss_reg + l.sum()
        total_loss = results['loss'] + loss_reg
        total_loss.backward()
        optimizer.step()
        return {'loss': total_loss.item(), 'acc': self.eval_fn(torch.argmax(results['preds'], -1), _curr_batch[-1].to(self.device))}
    
    def initTrainingStates(self, scenario, model, optimizer):
        return {'fishers': [], 'params': []}
    
    def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        """
            The event function to execute some processes after training the current task.
            
            EWC computes fisher information matrix and stores the learned weights to compute the penalty term in :func:`afterInference`
                
            Args:
                task_id (int): the index of the current task.
                curr_dataset (object): The dataset for the current task.
                curr_model (torch.nn.Module): the current trained model.
                curr_optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_training_states (dict): the dictionary containing the current training states.
        """
        super().processAfterTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)
        params = {name: torch.zeros_like(p) for name, p in curr_model.named_parameters()}
        fishers = {name: torch.zeros_like(p) for name, p in curr_model.named_parameters()}
        train_loader = self.prepareLoader(curr_dataset, curr_training_states)[0]
        
        total_num_items = 0
        for i, _curr_batch in enumerate(iter(train_loader)):
            curr_model.zero_grad()
            curr_results = self.inference(curr_model, _curr_batch, curr_training_states)
            curr_results['loss'].backward()
            curr_num_items =_curr_batch[1].shape[0]
            total_num_items += curr_num_items
            for name, p in curr_model.named_parameters():
                params[name] = p.data.clone().detach()
                fishers[name] += (p.grad.data.clone().detach() ** 2) * curr_num_items
                    
        for name, p in curr_model.named_parameters():
            fishers[name] /= total_num_items
                
        curr_training_states['fishers'].append(fishers)
        curr_training_states['params'].append(params)
    
class LCClassILEWCTrainer(LCTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        """
            EWC needs `lamb`, the additional hyperparamter for the regularization term used in :func:`afterInference`
        """
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.lamb = kwargs['lamb'] if 'lamb' in kwargs else 10000.
        
    def afterInference(self, results, model, optimizer, _curr_batch, training_states):
        """
            The event function to execute some processes right after the inference step (for training).
            We recommend performing backpropagation in this event function.
            
            EWC performs regularization process in this function.
            
            Args:
                results (dict): the returned dictionary from the event function `inference`.
                model (torch.nn.Module): the current trained model.
                optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
                
            Returns:
                A dictionary containing the information from the `results`.
        """
        loss_reg = 0
        for _param, _fisher in zip(training_states['params'], training_states['fishers']):
            for name, p in model.named_parameters():
                l = self.lamb * _fisher[name]
                l = l * ((p - _param[name]) ** 2)
                loss_reg = loss_reg + l.sum()
        total_loss = results['loss'] + loss_reg
        total_loss.backward()
        optimizer.step()
        return {'loss': total_loss.item(), 'acc': self.eval_fn(torch.argmax(results['preds'], -1), _curr_batch[-1].to(self.device))}
    
    def initTrainingStates(self, scenario, model, optimizer):
        return {'fishers': [], 'params': []}
    
    def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        """
            The event function to execute some processes after training the current task.
            
            EWC computes fisher information matrix and stores the learned weights to compute the penalty term in :func:`afterInference`
                
            Args:
                task_id (int): the index of the current task.
                curr_dataset (object): The dataset for the current task.
                curr_model (torch.nn.Module): the current trained model.
                curr_optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_training_states (dict): the dictionary containing the current training states.
        """
        super().processAfterTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)
        params = {name: torch.zeros_like(p) for name, p in curr_model.named_parameters()}
        fishers = {name: torch.zeros_like(p) for name, p in curr_model.named_parameters()}
        train_loader = self.prepareLoader(curr_dataset, curr_training_states)[0]
        
        total_num_items = 0
        for i, _curr_batch in enumerate(iter(train_loader)):
            curr_model.zero_grad()
            curr_results = self.inference(curr_model, _curr_batch, curr_training_states)
            curr_results['loss'].backward()
            curr_num_items =_curr_batch[1].shape[0]
            total_num_items += curr_num_items
            for name, p in curr_model.named_parameters():
                params[name] = p.data.clone().detach()
                fishers[name] += (p.grad.data.clone().detach() ** 2) * curr_num_items
                    
        for name, p in curr_model.named_parameters():
            fishers[name] /= total_num_items
                
        curr_training_states['fishers'].append(fishers)
        curr_training_states['params'].append(params)

class LCTimeILEWCTrainer(LCTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        """
            EWC needs `lamb`, the additional hyperparamter for the regularization term used in :func:`afterInference`
        """
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.lamb = kwargs['lamb'] if 'lamb' in kwargs else 10000.
        
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
        curr_model.observe_labels(torch.LongTensor([0]))
        self._reset_optimizer(curr_optimizer)
    
    def processEvalIteration(self, model, _curr_batch):
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
        results = self.inference(model, _curr_batch, None)
        return results['preds'], {'loss': results['loss'].item()}
    
    def afterInference(self, results, model, optimizer, _curr_batch, training_states):
        """
            The event function to execute some processes right after the inference step (for training).
            We recommend performing backpropagation in this event function.
            
            EWC performs regularization process in this function.
            
            Args:
                results (dict): the returned dictionary from the event function `inference`.
                model (torch.nn.Module): the current trained model.
                optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
                
            Returns:
                A dictionary containing the information from the `results`.
        """
        loss_reg = 0
        for _param, _fisher in zip(training_states['params'], training_states['fishers']):
            for name, p in model.named_parameters():
                l = self.lamb * _fisher[name]
                l = l * ((p - _param[name]) ** 2)
                loss_reg = loss_reg + l.sum()
        total_loss = results['loss'] + loss_reg
        total_loss.backward()
        optimizer.step()
        return {'loss': total_loss.item(), 'acc': self.eval_fn(results['preds'], _curr_batch[-1].to(self.device))}
    
    def initTrainingStates(self, scenario, model, optimizer):
        return {'fishers': [], 'params': []}
    
    def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        """
            The event function to execute some processes after training the current task.
            
            EWC computes fisher information matrix and stores the learned weights to compute the penalty term in :func:`afterInference`.
                
            Args:
                task_id (int): the index of the current task.
                curr_dataset (object): The dataset for the current task.
                curr_model (torch.nn.Module): the current trained model.
                curr_optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_training_states (dict): the dictionary containing the current training states.
        """
        super().processAfterTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)
        params = {name: torch.zeros_like(p) for name, p in curr_model.named_parameters()}
        fishers = {name: torch.zeros_like(p) for name, p in curr_model.named_parameters()}
        train_loader = self.prepareLoader(curr_dataset, curr_training_states)[0]
        
        total_num_items = 0
        for i, _curr_batch in enumerate(iter(train_loader)):
            curr_model.zero_grad()
            curr_results = self.inference(curr_model, _curr_batch, curr_training_states)
            curr_results['loss'].backward()
            curr_num_items =_curr_batch[1].shape[0]
            total_num_items += curr_num_items
            for name, p in curr_model.named_parameters():
                params[name] = p.data.clone().detach()
                fishers[name] += (p.grad.data.clone().detach() ** 2) * curr_num_items
                    
        for name, p in curr_model.named_parameters():
            fishers[name] /= total_num_items
                
        curr_training_states['fishers'].append(fishers)
        curr_training_states['params'].append(params)
        
class LPTimeILEWCTrainer(LPTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        """
            EWC needs `lamb`, the additional hyperparamter for the regularization term used in :func:`processTrainIteration.`
        """
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.lamb = kwargs['lamb'] if 'lamb' in kwargs else 1.
        self.T = kwargs['T'] if 'T' in kwargs else 2.
        
    def processTrainIteration(self, model, optimizer, _curr_batch, training_states):
        """
            The event function to handle every training iteration.
            
            EWC performs inference and regularization process in this function.
            
            Args:
                model (torch.nn.Module): the current trained model.
                optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
                
            Returns:
                A dictionary containing the outcomes (stats) during the training iteration.
        """
        graph, feats = map(lambda x: x.to(self.device), training_states['graph'])
        edges, labels = map(lambda x: x.to(self.device), _curr_batch)
        optimizer.zero_grad()
        srcs, dsts = edges[:, 0], edges[:, 1]
        neg_dsts = torch.randint(low=0, high=graph.num_nodes(), size=(srcs.shape[0],)).to(self.device)
        preds = model(graph, feats, srcs.repeat(2), torch.cat((edges[:, 1], neg_dsts), dim=0)).squeeze(-1)
        loss = self.loss_fn(preds, torch.cat((labels, torch.zeros_like(labels)), dim=0))
        
        loss_reg = 0
        for _param, _fisher in zip(training_states['params'], training_states['fishers']):
            for name, p in model.named_parameters():
                l = self.lamb * _fisher[name]
                l = l * ((p - _param[name]) ** 2)
                loss_reg = loss_reg + l.sum()
        loss = loss + loss_reg
        
        loss.backward()
        optimizer.step()
        return {'_num_items': preds.shape[0], 'loss': loss.item()}
    
    def initTrainingStates(self, scenario, model, optimizer):
        return {'fishers': [], 'params': []}
    
    def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        """
            The event function to execute some processes after training the current task.
            
            EWC computes fisher information matrix and stores the learned weights to compute the penalty term in :func:`processTrainIteration`.
                
            Args:
                task_id (int): the index of the current task.
                curr_dataset (object): The dataset for the current task.
                curr_model (torch.nn.Module): the current trained model.
                curr_optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_training_states (dict): the dictionary containing the current training states.
        """
        super().processAfterTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)
        params = {}
        fishers = {}
        train_loader = self.prepareLoader(curr_dataset, curr_training_states)[0]
        total_num_items = 0
        
        for i, _curr_batch in enumerate(iter(train_loader)):
            curr_model.zero_grad()
            graph, feats = map(lambda x: x.to(self.device), curr_training_states['graph'])
            edges, labels = map(lambda x: x.to(self.device), _curr_batch)
            srcs, dsts = edges[:, 0], edges[:, 1]
            neg_dsts = torch.randint(low=0, high=graph.num_nodes(), size=(srcs.shape[0],)).to(self.device)
            preds = curr_model(graph, feats, srcs.repeat(2), torch.cat((edges[:, 1], neg_dsts), dim=0)).squeeze(-1)
            loss = self.loss_fn(preds, torch.cat((labels, torch.zeros_like(labels)), dim=0))
            loss.backward()
            total_num_items += labels.shape[0]
            if i == 0:
                for name, p in curr_model.named_parameters():
                    params[name] = p.data.clone().detach()
                    fishers[name] = ((p.grad.data.clone().detach() ** 2) * labels.shape[0])
            else:
                for name, p in curr_model.named_parameters():
                    fishers[name] += ((p.grad.data.clone().detach() ** 2) * labels.shape[0])
                    
        for name, p in curr_model.named_parameters():
            fishers[name] /= total_num_items
        curr_training_states['fishers'].append(fishers)
        curr_training_states['params'].append(params)

class LPDomainILEWCTrainer(LPTimeILEWCTrainer):
    """
        This trainer has the same behavior as `LPTimeILEWCTrainer`.
    """
    pass