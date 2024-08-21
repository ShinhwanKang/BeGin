import sys
import torch
from begin.trainers.nodes import NCTrainer, NCMinibatchTrainer

class NCTaskILPIGNNTrainer(NCTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.retrain_beta = kwargs['retrain'] if 'retrain' in kwargs else 0.01
        self.num_memories = kwargs['num_memories'] if 'num_memories' in kwargs else 100
        self.num_memories = (self.num_memories // self.num_tasks)
        
    def processBeforeTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        n_hidden_before = (curr_model.n_hidden * task_id) // self.num_tasks
        n_hidden_after = (curr_model.n_hidden * (task_id + 1)) // self.num_tasks
        new_parameters = curr_model.expand_parameters(n_hidden_after - n_hidden_before, self.device)
        self.add_parameters(curr_model, curr_optimizer)
        super().processBeforeTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)

        # detect new classes
        if self.curr_task == 0:
            curr_model.class_to_task = curr_training_states.pop('class_to_task')
            curr_model.done_masks = []    
        new_classes = curr_dataset.ndata['task_specific_mask'][curr_dataset.ndata['train_mask']][0]
        curr_model.class_to_task[new_classes > 0] = self.curr_task
        
    def initTrainingStates(self, scenario, model, optimizer):
        return {'memories': [], 'class_to_task': -torch.ones(model.classifier.num_outputs, dtype=torch.long)}
        
    def beforeInference(self, model, optimizer, _curr_batch, training_states):
        for name, p in model.named_parameters():
            p.requires_grad_(False)
        for conv in model.convs:
            conv.weights[-1].requires_grad_(True)
            conv.norms[-1].requires_grad_(True)
        model.classifier.lins[-1].requires_grad_(True)
        
    def afterInference(self, results, model, optimizer, _curr_batch, training_states):
        """
            The event function to execute some processes right after the inference step (for training).
            We recommend performing backpropagation in this event function.
            
            ERGNN additionally computes the loss from the buffered nodes and applies it to backpropagation.
            
            Args:
                results (dict): the returned dictionary from the event function `inference`.
                model (torch.nn.Module): the current trained model.
                optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
                
            Returns:
                A dictionary containing the information from the `results`.
        """
        loss = results['loss']
        curr_batch, mask = _curr_batch
        if len(training_states['memories'])>0:
            # retrain phase
            buffered = torch.cat(training_states['memories'], dim=0)
            buffered_mask = torch.zeros(curr_batch.ndata['feat'].shape[0]).to(self.device)
            buffered_mask[buffered] = 1.
            buffered_mask = buffered_mask.to(torch.bool)
            buffered_loss = self.loss_fn(results['preds_full'][buffered_mask.cpu()].to(self.device), _curr_batch[0].ndata['label'][buffered_mask.cpu()].to(self.device))
            loss = loss + self.retrain_beta * buffered_loss
        loss.backward()
        optimizer.step()
        return {'loss': loss.item(),
                'acc': self.eval_fn(results['preds'].argmax(-1), curr_batch.ndata['label'][mask].to(self.device))}
        
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

    def inference(self, model, _curr_batch, training_states):
        """
            ERGNN requires node sampler. We use CM sampler as the default sampler.
            
            Args:
                scenario (begin.scenarios.common.BaseScenarioLoader): the given ScenarioLoader to the trainer
                model (torch.nn.Module): the given model to the trainer
                optmizer (torch.optim.Optimizer): the optimizer generated from the given `optimizer_fn` 
                
            Returns:
                Initialized training state (dict).
        """
        curr_batch, mask = _curr_batch
        preds = model(curr_batch.to(self.device), curr_batch.ndata['feat'].to(self.device), task_masks=curr_batch.ndata['task_specific_mask'].to(self.device))
        loss = self.loss_fn(preds[mask], curr_batch.ndata['label'][mask].to(self.device))
        return {'preds': preds[mask], 'loss': loss, 'preds_full': preds}
        
class NCClassILPIGNNTrainer(NCTrainer):
    """
        This trainer has the same behavior as `NCTrainer`.
    """
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.retrain_beta = kwargs['retrain'] if 'retrain' in kwargs else 0.01
        self.num_memories = kwargs['num_memories'] if 'num_memories' in kwargs else 100
        self.num_memories = (self.num_memories // self.num_tasks)
        
    def processBeforeTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        n_hidden_before = (curr_model.n_hidden * task_id) // self.num_tasks
        n_hidden_after = (curr_model.n_hidden * (task_id + 1)) // self.num_tasks
        new_parameters = curr_model.expand_parameters(n_hidden_after - n_hidden_before, self.device)
        self.add_parameters(curr_model, curr_optimizer)
        super().processBeforeTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)

    def beforeInference(self, model, optimizer, _curr_batch, training_states):
        for name, p in model.named_parameters():
            p.requires_grad_(False)
        for conv in model.convs:
            conv.weights[-1].requires_grad_(True)
            conv.norms[-1].requires_grad_(True)
        model.classifier.lins[-1].requires_grad_(True)

    def initTrainingStates(self, scenario, model, optimizer):
        return {'memories': []}

    def inference(self, model, _curr_batch, training_states):
        curr_batch, mask = _curr_batch
        preds = model(curr_batch.to(self.device), curr_batch.ndata['feat'].to(self.device))
        loss = self.loss_fn(preds[mask], curr_batch.ndata['label'][mask].to(self.device))
        return {'preds': preds[mask], 'loss': loss, 'preds_full': preds}
        
    def afterInference(self, results, model, optimizer, _curr_batch, training_states):
        """
            The event function to execute some processes right after the inference step (for training).
            We recommend performing backpropagation in this event function.
            
            ERGNN additionally computes the loss from the buffered nodes and applies it to backpropagation.
            
            Args:
                results (dict): the returned dictionary from the event function `inference`.
                model (torch.nn.Module): the current trained model.
                optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
                
            Returns:
                A dictionary containing the information from the `results`.
        """
        loss = results['loss']
        curr_batch, mask = _curr_batch
        if len(training_states['memories'])>0:
            # retrain phase
            buffered = torch.cat(training_states['memories'], dim=0)
            buffered_mask = torch.zeros(curr_batch.ndata['feat'].shape[0]).to(self.device)
            buffered_mask[buffered] = 1.
            buffered_mask = buffered_mask.to(torch.bool)
            buffered_loss = self.loss_fn(results['preds_full'][buffered_mask.cpu()].to(self.device), _curr_batch[0].ndata['label'][buffered_mask.cpu()].to(self.device))
            loss = loss + self.retrain_beta * buffered_loss
        loss.backward()
        optimizer.step()
        return {'loss': loss.item(),
                'acc': self.eval_fn(results['preds'].argmax(-1), curr_batch.ndata['label'][mask].to(self.device))}
        
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
        
class NCClassILPIGNNMinibatchTrainer(NCMinibatchTrainer):
    """
        This trainer has the same behavior as `NCMinibatchTrainer`.
    """
    pass

class NCDomainILPIGNNTrainer(NCClassILPIGNNTrainer):
    """
        This trainer has the same behavior as `NCTrainer`.
    """
    pass
        
class NCTimeILPIGNNTrainer(NCTrainer):
    """
        This trainer has the same behavior as `NCTrainer`.
    """
    pass