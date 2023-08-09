import torch
import copy
import torch.nn.functional as F
from begin.trainers.graphs import GCTrainer

class GCTaskILLwFTrainer(GCTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        """
            LwF needs additional hyperparamters, lamb and T, for knowledge distillation process in :func:`afterInference`.
        """
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        # additional hyperparameters
        self.lamb = kwargs['lamb'] if 'lamb' in kwargs else 1.
        self.T = kwargs['T'] if 'T' in kwargs else 2.
    
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
        graphs, labels, _ = _curr_batch
        if 'prev_model' in training_states:
            # apply Knowledge distillation for every previous task
            for tid in range(training_states['task_id']):
                observed_labels = model.get_observed_labels(tid)
                task_specific_mask = observed_labels.unsqueeze(0).repeat(graphs.batch_size, 1)
                task_specific_batch = (graphs, labels, task_specific_mask)
                prv_results = self.inference(training_states['prev_model'], task_specific_batch, training_states)
                curr_results = self.inference(model, task_specific_batch, training_states)
                curr_kd_loss = F.softmax(prv_results['preds'] / self.T, dim=-1)
                curr_kd_loss = curr_kd_loss * F.log_softmax(curr_results['preds'] / self.T, dim=-1)
                curr_kd_loss[..., ~observed_labels] = 0.
                kd_loss = kd_loss - curr_kd_loss.sum(-1).mean()
            
        total_loss = results['loss'] + (self.lamb * kd_loss)
        total_loss.backward()
        optimizer.step()
        return {'_num_items': results['preds'].shape[0],
                'loss': total_loss.item(),
                'acc': self.eval_fn(self.predictionFormat(results), labels.to(self.device))}
    
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

class GCClassILLwFTrainer(GCTrainer):
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
            # apply Knowledge distillation from the previously learned weights
            prv_results = self.inference(training_states['prev_model'], _curr_batch, training_states)
            observed_labels = training_states['prev_observed_labels']
            kd_loss = F.softmax(prv_results['preds'][..., observed_labels].detach() / self.T, dim=-1)
            kd_loss = kd_loss * F.log_softmax(results['preds'][..., observed_labels] / self.T, dim=-1)
            kd_loss = (-kd_loss).sum(-1).mean()    
        total_loss = results['loss'] + (self.lamb * kd_loss)
        total_loss.backward()
        optimizer.step()
        return {'_num_items': results['preds'].shape[0], 'loss': total_loss.item(), 'acc': self.eval_fn(self.predictionFormat(results), _curr_batch[1].to(self.device))}
    
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
        # save the current model (to use KD for future tasks)
        curr_training_states['prev_model'] = copy.deepcopy(curr_model)
        curr_training_states['prev_observed_labels'] = curr_model.get_observed_labels().clone().detach()

class GCDomainILLwFTrainer(GCClassILLwFTrainer):
    """
        This trainer has the same behavior as `GCClassILLwFTrainer`.
    """
    pass
        
class GCTimeILLwFTrainer(GCClassILLwFTrainer):
    """
        This trainer has the same behavior as `GCClassILLwFTrainer`.
    """
    pass