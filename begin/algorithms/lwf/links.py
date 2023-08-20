import sys
import numpy as np
import torch
import copy, dgl
import torch.nn.functional as F
from begin.trainers.links import LCTrainer, LPTrainer

class LCTaskILLwFTrainer(LCTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        """
            LwF needs additional hyperparamters, lamb and T, for knowledge distillation process in :func:`afterInference`.
        """
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        # additional hyperparameters
        self.lamb = kwargs['lamb'] if 'lamb' in kwargs else 1.
        self.T = kwargs['T'] if 'T' in kwargs else 2.
        
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
            
            LwF performs knowledge distillation process in this function.
            
            Args:
                results (dict): the returned dictionary from the event function `inference`.
                model (torch.nn.Module): the current trained model.
                optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
                
            Returns:
                A dictionary containing the information from the `results`.
        """
        kd_loss = 0.
        if 'prev_model' in training_states:
            # apply Knowledge distillation from the previously learned weights
            for tid in range(training_states['task_id']):
                task_specific_batch = copy.deepcopy(_curr_batch)
                observed_labels = model.get_observed_labels(tid)
                task_specific_masks = observed_labels.unsqueeze(0).repeat(_curr_batch[-1].shape[0], 1)
                
                curr_batch, srcs, dsts, _, labels = _curr_batch
                task_specific_batch = (curr_batch, srcs, dsts, task_specific_masks, labels)
                prv_results = self.inference(training_states['prev_model'], task_specific_batch, training_states)
                curr_results = self.inference(model, task_specific_batch, training_states)
                curr_kd_loss = F.softmax(prv_results['preds'] / self.T, dim=-1)
                curr_kd_loss = curr_kd_loss * F.log_softmax(curr_results['preds'] / self.T, dim=-1)
                curr_kd_loss[..., ~observed_labels] = 0.
                kd_loss = kd_loss - curr_kd_loss.sum(-1).mean()
        total_loss = results['loss'] + (self.lamb * kd_loss)
        total_loss.backward()
        optimizer.step()
        return {'loss': total_loss.item(), 'acc': self.eval_fn(torch.argmax(results['preds'], -1), _curr_batch[-1].to(self.device))}
    
    def processBeforeTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        """
            The event function to execute some processes before training.
        
            We need to store previously learned weights for the knowledge distillation process in :func:`afterInference`.
        
            Args:
                task_id (int): the index of the current task
                curr_dataset (object): The dataset for the current task.
                curr_model (torch.nn.Module): the current trained model.
                curr_optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_training_states (dict): the dictionary containing the current training states.
        """
        super().processBeforeTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)
        # save the current model (to use KD for future tasks)
        curr_training_states['task_id'] = task_id
        curr_training_states['prev_model'] = copy.deepcopy(curr_model)
    
class LCClassILLwFTrainer(LCTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        """
            LwF needs additional hyperparamters, lamb and T, for knowledge distillation process in :func:`afterInference`.
        """
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        # additional hyperparameters
        self.lamb = kwargs['lamb'] if 'lamb' in kwargs else 1.
        self.T = kwargs['T'] if 'T' in kwargs else 2.
        
    def afterInference(self, results, model, optimizer, _curr_batch, training_states):
        """
            The event function to execute some processes right after the inference step (for training).
            We recommend performing backpropagation in this event function.
            
            LwF performs knowledge distillation process in this function.
            
            Args:
                results (dict): the returned dictionary from the event function `inference`.
                model (torch.nn.Module): the current trained model.
                optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
                
            Returns:
                A dictionary containing the information from the `results`.
        """
        kd_loss = 0.
        if 'prev_model' in training_states:
            prv_results = self.inference(training_states['prev_model'], _curr_batch, training_states)
            observed_labels = training_states['prev_observed_labels']
            kd_loss = F.softmax(prv_results['preds'][..., observed_labels].detach() / self.T, dim=-1)
            kd_loss = kd_loss * F.log_softmax(results['preds'][..., observed_labels] / self.T, dim=-1)
            kd_loss = (-kd_loss).sum(-1).mean()
        total_loss = results['loss'] + (self.lamb * kd_loss)
        total_loss.backward()
        optimizer.step()
        return {'loss': total_loss.item(), 'acc': self.eval_fn(torch.argmax(results['preds'], -1), _curr_batch[-1].to(self.device))}
    
    def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        """
            The event function to execute some processes after training.
        
            We need to store previously learned weights for the knowledge distillation process in :func:`afterInference`.
        
            Args:
                task_id (int): the index of the current task
                curr_dataset (object): The dataset for the current task.
                curr_model (torch.nn.Module): the current trained model.
                curr_optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_training_states (dict): the dictionary containing the current training states.
        """
        curr_model.load_state_dict(curr_training_states['best_weights'])
        curr_training_states['prev_model'] = copy.deepcopy(curr_model)
        curr_training_states['prev_observed_labels'] = curr_model.get_observed_labels().clone().detach()

class LCTimeILLwFTrainer(LCTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        """
            LwF needs additional hyperparamters, lamb and T, for knowledge distillation process in :func:`afterInference`.
        """
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.lamb = kwargs['lamb'] if 'lamb' in kwargs else 1.
        self.T = kwargs['T'] if 'T' in kwargs else 2.
        
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
            The event function to handle every evaluation iteration.
            We need to extend the base function since the output format is slightly different from the base trainer.
            
            Args:
                model (torch.nn.Module): the current trained model.
                curr_batch (object): the data (or minibatch) for the current iteration.
                
            Returns:
                A dictionary containing the outcomes (stats) during the evaluation iteration.
        """
        results = self.inference(model, _curr_batch, None)
        return results['preds'], {'loss': results['loss'].item()}
    
    def afterInference(self, results, model, optimizer, _curr_batch, training_states):
        """
            The event function to execute some processes right after the inference step (for training).
            We recommend performing backpropagation in this event function.
            
            LwF performs knowledge distillation process in this function.
            
            Args:
                results (dict): the returned dictionary from the event function `inference`.
                model (torch.nn.Module): the current trained model.
                optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
                
            Returns:
                A dictionary containing the information from the `results`.
        """
        kd_loss = 0.
        if 'prev_model' in training_states:
            prv_results = self.inference(training_states['prev_model'], _curr_batch, training_states)
            observed_labels = training_states['prev_observed_labels']
            kd_loss = F.sigmoid(prv_results['preds'][..., observed_labels].detach() / 2 / self.T)
            kd_loss = kd_loss * F.logsigmoid(results['preds'][..., observed_labels] / 2 / self.T)
            kd_loss = (-kd_loss).sum(-1).mean()
        total_loss = results['loss'] + (self.lamb * kd_loss)
        total_loss.backward()
        optimizer.step()
        return {'loss': total_loss.item(), 'acc': self.eval_fn(results['preds'], _curr_batch[-1].to(self.device))}
    
    def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        """
            The event function to execute some processes after training.
        
            We need to store previously learned weights for the knowledge distillation process in :func:`afterInference`.
        
            Args:
                task_id (int): the index of the current task
                curr_dataset (object): The dataset for the current task.
                curr_model (torch.nn.Module): the current trained model.
                curr_optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_training_states (dict): the dictionary containing the current training states.
        """
        
        curr_model.load_state_dict(curr_training_states['best_weights'])
        curr_training_states['prev_model'] = copy.deepcopy(curr_model)
        curr_training_states['prev_observed_labels'] = curr_model.get_observed_labels().clone().detach()

class LPTimeILLwFTrainer(LPTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        """
            LwF needs additional hyperparamters, lamb and T, for knowledge distillation process in :func:`afterInference`.
        """
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.lamb = kwargs['lamb'] if 'lamb' in kwargs else 1.
        self.T = kwargs['T'] if 'T' in kwargs else 2.
        
    def processTrainIteration(self, model, optimizer, _curr_batch, training_states):
        """
            The event function to handle every training iteration.
            
            LwF performs inference and knowledge distillation process in this function.
            
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
        labels = labels.to(self.device)
        loss = self.loss_fn(preds, torch.cat((labels, torch.zeros_like(labels)), dim=0))
        kd_loss = 0.
        if 'prev_model' in training_states:
            prev_model = training_states['prev_model']
            prv_preds = prev_model(graph, feats, srcs.repeat(2), torch.cat((edges[:, 1], neg_dsts), dim=0)).squeeze(-1).detach()
            kd_loss = F.sigmoid((prv_preds / 2) / self.T) * F.logsigmoid((preds / 2) / self.T)
            kd_loss = kd_loss + F.sigmoid((-prv_preds / 2) / self.T) * F.logsigmoid((-preds / 2) / self.T)
            kd_loss = (-kd_loss).mean()
            
        loss = loss + (self.lamb * kd_loss)
        loss.backward()
        optimizer.step()
        return {'_num_items': preds.shape[0], 'loss': loss.item()}
    
    def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        """
            The event function to execute some processes after training the current task.
            
            We need to store previously learned weights for the knowledge distillation process in :func:`processTrainIteration`.
                
            Args:
                task_id (int): the index of the current task.
                curr_dataset (object): The dataset for the current task.
                curr_model (torch.nn.Module): the current trained model.
                curr_optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_training_states (dict): the dictionary containing the current training states.
        """
        curr_model.load_state_dict(curr_training_states['best_weights'])
        curr_training_states['prev_model'] = copy.deepcopy(curr_model)
        
class LPDomainILLwFTrainer(LPTimeILLwFTrainer):
    """
        This trainer has the same behavior as `LPTimeILLwFTrainer`.
    """
    pass