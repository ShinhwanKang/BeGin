import sys
import torch
import dgl
from begin.trainers.nodes import NCTrainer, NCMinibatchTrainer
import copy
import dgl.function as fn

class NCTaskILPIGNNTrainer(NCTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        """
            `num_memories` is the hyperparameter for size of the memory.
            `retrain_beta` is the hyperparameter for handling the size imbalance problem in the parameter-isolation phase.
        """
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.retrain_beta = kwargs['retrain'] if 'retrain' in kwargs else 0.01
        self.num_memories = kwargs['num_memories'] if 'num_memories' in kwargs else 100
        self.num_memories = (self.num_memories // self.num_tasks)
        
    def processBeforeTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        """
            PI-GNN requires extending the network before running each task.

            Args:
                task_id (int): the index of the current task
                curr_dataset (object): The dataset for the current task.
                curr_model (torch.nn.Module): the current trained model.
                curr_optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_training_states (dict): the dictionary containing the current training states.
        """
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
        """
            The event function to execute some processes right before inference (for training).
            
            Before training, PI-GNN needs to freeze parameters from the past tasks in parameter-isolation phase.
            
            Args:
                model (torch.nn.Module): the current trained model.
                optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
        """
        model.train()
        model.requires_grad_(False)
        for conv in model.convs:
            conv.weights[-1].requires_grad_(True)
            for norm in conv.norms[:-1]:
                norm.eval()
            conv.norms[-1].train()
            conv.norms[-1].requires_grad_(True)
        model.classifier.lins[-1].requires_grad_(True)
        model.curr_task = -1
        
    def afterInference(self, results, model, optimizer, _curr_batch, training_states):
        """
            The event function to execute some processes right after the inference step (for training).
            We recommend performing backpropagation in this event function.
            
            PI-GNN additionally computes the loss from the buffered nodes and applies it to backpropagation in parameter-isolation phase.
            
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

            PI-GNN samples the instances in the training dataset for the future tasks.
                
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
        curr_batch, mask = _curr_batch
        preds = model(curr_batch.to(self.device), curr_batch.ndata['feat'].to(self.device), task_masks=curr_batch.ndata['task_specific_mask'].to(self.device))
        loss = self.loss_fn(preds[mask], curr_batch.ndata['label'][mask].to(self.device))
        return {'preds': preds[mask], 'loss': loss, 'preds_full': preds}

    def processEvalIteration(self, model, _curr_batch):
        """
            The event function to handle every evaluation iteration.
            
            PI-GNN has to use different masks to evaluate the performance for each task.
            
            Args:
                model (torch.nn.Module): the current trained model.
                curr_batch (object): the data (or minibatch) for the current iteration.
                task_id (int): the id of a task
                
            Returns:
                A dictionary containing the outcomes (stats) during the evaluation iteration.
        """

        curr_batch, mask = _curr_batch
        task_ids = torch.max(curr_batch.ndata['task_specific_mask'][mask] * model.class_to_task, dim=-1).values
        task_checks = torch.min(curr_batch.ndata['task_specific_mask'][mask] * model.class_to_task, dim=-1).values
        test_nodes = torch.arange(mask.shape[0])[mask]
        task_ids[task_checks < 0] = self.curr_task
        
        num_samples = torch.bincount(task_ids.detach().cpu(), minlength=self.curr_task+1)
        total_results = torch.zeros_like(test_nodes).to(self.device)
        total_loss = 0.
        # handle the samples with different tasks separately
        for i in range(self.curr_task, -1, -1):
            if num_samples[i].item() == 0: continue
            model.curr_task = i
            eval_mask = torch.zeros(mask.shape[0])
            eval_mask[test_nodes[task_ids == i]] = 1
            results = self.inference(model, (curr_batch, eval_mask.bool()), None)
            total_results[task_ids == i] = torch.argmax(results['preds'], dim=-1)
            total_loss += results['loss'].item() * num_samples[i].item()
        total_loss /= torch.sum(num_samples).item()
        return total_results, {'loss': total_loss}
        
class NCClassILPIGNNTrainer(NCTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        """
            `num_memories` is the hyperparameter for size of the memory.
            `retrain_beta` is the hyperparameter for handling the size imbalance problem in the parameter-isolation phase.
        """
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.retrain_beta = kwargs['retrain'] if 'retrain' in kwargs else 0.01
        self.num_memories = kwargs['num_memories'] if 'num_memories' in kwargs else 100
        self.num_memories = (self.num_memories // self.num_tasks)
        
    def processBeforeTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        """
            PI-GNN requires extending the network before running each task.

            Args:
                task_id (int): the index of the current task
                curr_dataset (object): The dataset for the current task.
                curr_model (torch.nn.Module): the current trained model.
                curr_optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_training_states (dict): the dictionary containing the current training states.
        """
        n_hidden_before = (curr_model.n_hidden * task_id) // self.num_tasks
        n_hidden_after = (curr_model.n_hidden * (task_id + 1)) // self.num_tasks
        new_parameters = curr_model.expand_parameters(n_hidden_after - n_hidden_before, self.device)
        self.add_parameters(curr_model, curr_optimizer)
        super().processBeforeTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)

    def beforeInference(self, model, optimizer, _curr_batch, training_states):
        """
            The event function to execute some processes right before inference (for training).
            
            Before training, PI-GNN needs to freeze parameters from the past tasks in parameter-isolation phase.
            
            Args:
                model (torch.nn.Module): the current trained model.
                optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
        """
        model.train()
        model.requires_grad_(False)
        for conv in model.convs:
            conv.weights[-1].requires_grad_(True)
            for norm in conv.norms[:-1]:
                norm.eval()
            conv.norms[-1].train()
            conv.norms[-1].requires_grad_(True)
        model.classifier.lins[-1].requires_grad_(True)
        model.curr_task = -1
    
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
            
            PI-GNN additionally computes the loss from the buffered nodes and applies it to backpropagation in parameter-isolation phase.
            
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
                'acc': self.eval_fn(self.predictionFormat(results), curr_batch.ndata['label'][mask].to(self.device))}
        
    def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        """
            The event function to execute some processes after training the current task.

            PI-GNN samples the instances in the training dataset for the future tasks.
                
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
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        """
            `num_memories` is the hyperparameter for size of the memory.
            `retrain_beta` is the hyperparameter for handling the size imbalance problem in the parameter-isolation phase.
        """
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.retrain_beta = kwargs['retrain'] if 'retrain' in kwargs else 0.01
        self.num_memories = kwargs['num_memories'] if 'num_memories' in kwargs else 100
        self.num_memories = (self.num_memories // self.num_tasks)
        
    def processBeforeTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        """
            PI-GNN requires extending the network before running each task.

            Args:
                task_id (int): the index of the current task
                curr_dataset (object): The dataset for the current task.
                curr_model (torch.nn.Module): the current trained model.
                curr_optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_training_states (dict): the dictionary containing the current training states.
        """
        n_hidden_before = (curr_model.n_hidden * task_id) // self.num_tasks
        n_hidden_after = (curr_model.n_hidden * (task_id + 1)) // self.num_tasks
        new_parameters = curr_model.expand_parameters(n_hidden_after - n_hidden_before, self.device)
        self.add_parameters(curr_model, curr_optimizer)
        super().processBeforeTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)

    def beforeInference(self, model, optimizer, _curr_batch, training_states):
        """
            The event function to execute some processes right before inference (for training).
            
            Before training, PI-GNN needs to freeze parameters from the past tasks in parameter-isolation phase.
            
            Args:
                model (torch.nn.Module): the current trained model.
                optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
        """
        model.train()
        model.requires_grad_(False)
        for conv in model.convs:
            conv.weights[-1].requires_grad_(True)
            for norm in conv.norms[:-1]:
                norm.eval()
            conv.norms[-1].train()
            conv.norms[-1].requires_grad_(True)
        model.classifier.lins[-1].requires_grad_(True)
        model.curr_task = -1
    
    def initTrainingStates(self, scenario, model, optimizer):
        return {'memories': []}

    def afterInference(self, results, model, optimizer, _curr_batch, training_states):
        """
            The event function to execute some processes right after the inference step (for training).
            We recommend performing backpropagation in this event function.
            
            PI-GNN additionally computes the loss from the buffered nodes and applies it to backpropagation in parameter-isolation phase.
            
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
        if len(training_states['memories'])>0:
            buffered_loss = 0.
            for tid in range(len(training_states['memories'])):
                _buffered_task_loss = 0.
                for _buf_batch in training_states['memories'][tid]:
                    buf_results = self.inference(model, _buf_batch, training_states)
                    _buffered_task_loss = _buffered_task_loss + buf_results['loss']
                buffered_loss = buffered_loss + (_buffered_task_loss / len(training_states['memories'][tid]))
            buffered_loss = buffered_loss / len(training_states['memories'])    
            loss = loss + self.retrain_beta * buffered_loss
        loss.backward()
        optimizer.step()
        return {'loss': loss.item(),
                'acc': self.eval_fn(self.predictionFormat(results), _curr_batch[-1][-1].dstdata['label'].to(self.device))}
        
    def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        """
            The event function to execute some processes after training the current task.

            PI-GNN samples the instances in the training dataset for the future tasks.
                
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
        train_loader = dgl.dataloading.DataLoader(
            curr_dataset, candidates[perm[:self.num_memories]], train_sampler,
            batch_size=131072,
            shuffle=True,
            drop_last=False,
            num_workers=1, worker_init_fn=self._dataloader_seed_worker, generator=g_train)
        
        curr_training_states['memories'].append(train_loader)
        
class NCDomainILPIGNNTrainer(NCClassILPIGNNTrainer):
    """
        This trainer has the same behavior as `NCClassILPIGNNTrainer`.
    """
    pass
        
class NCTimeILPIGNNTrainer(NCClassILPIGNNTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        """
            `num_memories` is the hyperparameter for size of the memory.
            `retrain_beta` is the hyperparameter for handling the size imbalance problem in the parameter-isolation phase.
        """
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.retrain_beta = kwargs['retrain'] if 'retrain' in kwargs else 0.01
        self.num_memories = kwargs['num_memories'] if 'num_memories' in kwargs else 100
        self.num_memories = (self.num_memories // self.num_tasks)
        
    def processBeforeTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        """
            PI-GNN requires extending the network before running each task.

            Args:
                task_id (int): the index of the current task
                curr_dataset (object): The dataset for the current task.
                curr_model (torch.nn.Module): the current trained model.
                curr_optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_training_states (dict): the dictionary containing the current training states.
        """
        n_hidden_before = (curr_model.n_hidden * task_id) // self.num_tasks
        n_hidden_after = (curr_model.n_hidden * (task_id + 1)) // self.num_tasks
        new_parameters = curr_model.expand_parameters(n_hidden_after - n_hidden_before, self.device)
        self.add_parameters(curr_model, curr_optimizer)
        super().processBeforeTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)
        
        curr_training_states['task_id'] = task_id
        curr_training_states['n_epochs'] = 0
        if task_id == 0: curr_training_states['phase'] = 'retrain'
        else: curr_training_states['phase'] = 'rectify'
            
        if task_id == 0:
            curr_training_states['prv_degs'] = torch.zeros_like(curr_dataset.in_degrees())
        else:
            with curr_dataset.local_scope():
                new_degs = curr_dataset.in_degrees()
                curr_dataset.ndata['changed'] = (curr_training_states['prv_degs'] != new_degs).float()
                for i in range(curr_model.n_layers):
                    curr_dataset.update_all(fn.copy_u('changed', 'm'), fn.max('m', 'changed'))
                    curr_dataset.ndata['changed'][new_degs == 0] = 0.           
                curr_training_states['unchanged'] = torch.cat(curr_training_states['memories'], dim=-1)[((curr_dataset.ndata['changed'] < 0.5) & (new_degs > 0))[torch.cat(curr_training_states['memories'], dim=-1)]]
                
    def beforeInference(self, model, optimizer, _curr_batch, curr_training_states):
        """
            The event function to execute some processes right before inference (for training).
            
            Before training, PI-GNN needs to freeze parameters from the past tasks in parameter-isolation phase.
            
            Args:
                model (torch.nn.Module): the current trained model.
                optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
        """
        if curr_training_states['phase'] == 'retrain':
            model.train()
            model.requires_grad_(False)
            for conv in model.convs:
                conv.weights[-1].requires_grad_(True)
                for norm in conv.norms[:-1]:
                    norm.eval()
                conv.norms[-1].train()
                conv.norms[-1].requires_grad_(True)
            model.classifier.lins[-1].requires_grad_(True)
            model.curr_task = -1
        else:
            model.train()
            model.requires_grad_(True)
            model.curr_task = curr_training_states['task_id'] - 1

    def afterInference(self, results, model, optimizer, _curr_batch, curr_training_states):
        """
            The event function to execute some processes right after the inference step (for training).
            We recommend performing backpropagation in this event function.
            
            PI-GNN additionally computes the loss from the buffered nodes and applies it to backpropagation.
            
            Args:
                results (dict): the returned dictionary from the event function `inference`.
                model (torch.nn.Module): the current trained model.
                optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
                
            Returns:
                A dictionary containing the information from the `results`.
        """
        if curr_training_states['phase'] == 'retrain':
            loss = results['loss']
            curr_batch, mask = _curr_batch
            if len(curr_training_states['memories'])>0:
                # retrain phase
                buffered = torch.cat(curr_training_states['memories'], dim=0)
                buffered_mask = torch.zeros(curr_batch.ndata['feat'].shape[0]).to(self.device)
                buffered_mask[buffered] = 1.
                buffered_mask = buffered_mask.to(torch.bool)
                buffered_loss = self.loss_fn(results['preds_full'][buffered_mask.cpu()].to(self.device), _curr_batch[0].ndata['label'][buffered_mask.cpu()].to(self.device))
                loss = loss + self.retrain_beta * buffered_loss
            loss.backward()
            optimizer.step()
            return {'loss': loss.item(),
                    'acc': self.eval_fn(self.predictionFormat(results), curr_batch.ndata['label'][mask].to(self.device))}
        else:
            buffered_loss = self.loss_fn(results['preds_full'][curr_training_states['unchanged']].to(self.device), _curr_batch[0].ndata['label'][curr_training_states['unchanged']].to(self.device))
            loss = buffered_loss
            loss.backward()
            optimizer.step()
            return {'loss': loss.item(), 'acc': loss.item()} # self.eval_fn(self.predictionFormat(results), curr_batch.ndata['label'][mask].to(self.device))}
            
    def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        """
            The event function to execute some processes after training the current task.

            PI-GNN samples the instances in the training dataset for the future tasks.
                
            Args:
                task_id (int): the index of the current task.
                curr_dataset (object): The dataset for the current task.
                curr_model (torch.nn.Module): the current trained model.
                curr_optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_training_states (dict): the dictionary containing the current training states.
        """
        super().processAfterTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)
        curr_training_states['prv_degs'] = curr_dataset.in_degrees()

    def processAfterEachIteration(self, curr_model, curr_optimizer, curr_training_states, curr_iter_results):
        """
            The event function to execute some processes for every end of each epoch.
            Whether to continue training or not is determined by the return value of this function.
            If the returned value is False, the trainer stops training the current model in the current task.
            For PI-GNN, if the rectify phase ends, we need to move to the parameter-isolation phase.
            
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
        if curr_training_states['phase'] == 'retrain':
            curr_training_states['n_epochs'] += 1
            val_loss = curr_iter_results['val_stats']['loss']
            # maintain the best parameter
            if val_loss < curr_training_states['best_val_loss']:
                curr_training_states['best_val_loss'] = val_loss
                curr_training_states['best_weights'] = copy.deepcopy(curr_model.state_dict())
            
            # integration with scheduler
            scheduler = curr_training_states['scheduler']
            scheduler.step(val_loss)
            
            # stopping criteria for training
            if -1e-9 < (curr_optimizer.param_groups[0]['lr'] - scheduler.min_lrs[0]) < 1e-9:
                # earlystopping!
                return False
            return True
        else:
            curr_training_states['n_epochs'] += 1
            val_loss = curr_iter_results['train_stats']['loss']
            # maintain the best parameter
            if val_loss < curr_training_states['best_val_loss']:
                curr_training_states['best_val_loss'] = val_loss
                curr_training_states['best_weights'] = copy.deepcopy(curr_model.state_dict())
            
            # integration with scheduler
            scheduler = curr_training_states['scheduler']
            scheduler.step(val_loss)
            
            # stopping criteria for training
            if (-1e-9 < (curr_optimizer.param_groups[0]['lr'] - scheduler.min_lrs[0]) < 1e-9) or (curr_training_states['n_epochs'] >= (self.max_num_epochs // 10)):
                curr_training_states['phase'] = 'retrain'
                curr_model.load_state_dict(curr_training_states['best_weights'])
                self._reset_optimizer(curr_optimizer)
                curr_training_states['scheduler'] = self.scheduler_fn(curr_optimizer)
                curr_training_states['best_val_loss'] = 1e10
            return True