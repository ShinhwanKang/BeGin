import sys
import numpy as np
import torch
from copy import deepcopy
import copy, dgl
import torch.nn.functional as F
from begin.trainers.links import LCTrainer, LPTrainer

class LCTaskILTWPTrainer(LCTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        """
            TWP needs three additional parameters `lambda_l`, `lambda_t`, and `beta`.
            `lambda_l` is the hyperparamter for the regularization term (similar to EWC) used in :func:`afterInference`.
            `lambda_t` is the hyperparamter for the regularization term (with topological information) used in :func:`afterInference`.
            `beta` is the hyperparameter for the regularization term related to `cur_importance_score` in :func:`afterInference`.
        """
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.lambda_l = kwargs['lambda_l'] if 'lambda_l' in kwargs else 10000
        self.lambda_t = kwargs['lambda_t']  if 'lambda_t' in kwargs else 10000
        self.beta = kwargs['beta'] if 'beta' in kwargs else 0.1
        
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
    
    def inference(self, model, _curr_batch, training_states, return_elist=False):
        """
            The event function to execute inference step.
        
            For task-IL, we need to additionally consider task information for the inference step.
            TWP requires edge weights computed by attention mechanism.
        
            Args:
                model (torch.nn.Module): the current trained model.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
                
            Returns:
                A dictionary containing the inference results, such as prediction result and loss.
        """
        curr_batch, srcs, dsts, task_masks, labels = _curr_batch
        preds = model(curr_batch.to(self.device), curr_batch.ndata['feat'].to(self.device), srcs, dsts, return_elist=return_elist, task_masks=task_masks)
        if return_elist: preds, elist = preds
        loss = self.loss_fn(preds, labels.to(self.device))
        return {'preds': preds, 'loss': loss, 'elist': elist if return_elist else None}
    
    def afterInference(self, results, model, optimizer, _curr_batch, training_states): 
        """
            The event function to execute some processes right after the inference step (for training).
            We recommend performing backpropagation in this event function.
        
            TWP performs regularization process in this function.
        
            Args:
                results (dict): the returned dictionary from the event function `inference`.
                model (torch.nn.Module): the current trained model.
                optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
                
            Returns:
                A dictionary containing the information from the `results`.
        """
        cls_loss = results['loss']
        cls_loss.backward(retain_graph=True)
        cur_importance_score = 0.
        for gs in model.parameters():
            cur_importance_score += torch.norm(gs.grad.data.clone(), p=1) 
        old_importance_score = 0.
        for tt in range(0, training_states['current_task']):
            for i, p in enumerate(model.parameters()):
                I_n = self.lambda_l * training_states['cls_important_score'][tt][i] + self.lambda_t * training_states['topology_important_score'][tt][i]
                I_n = I_n * (p - training_states['optpar'][tt][i]).pow(2)
                old_importance_score += I_n.sum()            
        cls_loss += old_importance_score + self.beta * cur_importance_score
        cls_loss.backward()
        optimizer.step()
        return {'loss': results['loss'].item(), 'acc': self.eval_fn(torch.argmax(results['preds'], dim=-1), _curr_batch[-1].to(self.device))}
    
    def initTrainingStates(self, scenario, model, optimizer):
        return {'current_task':0, 'fisher_loss':{}, 'fisher_att':{}, 'optpar':{}, 'mem_mask':None, 'cls_important_score':{}, 'topology_important_score':{}}
    
    def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):        
        """
            The event function to execute some processes after training the current task.

            TWP computes weights for regularization process and stores the learned weights in this function.
                
            Args:
                task_id (int): the index of the current task.
                curr_dataset (object): The dataset for the current task.
                curr_model (torch.nn.Module): the current trained model.
                curr_optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_training_states (dict): the dictionary containing the current training states.
        """
        curr_model.load_state_dict(curr_training_states['best_weights'])
        optpars = [None for (name, p) in curr_model.named_parameters()]
        cls_scores = [torch.zeros_like(p.data) for (name, p) in curr_model.named_parameters()]
        topology_scores = [torch.zeros_like(p.data) for (name, p) in curr_model.named_parameters()]
        
        train_loader = self.prepareLoader(curr_dataset, curr_training_states)[0]
        total_num_items = 0
        for i, _curr_batch in enumerate(iter(train_loader)):
            curr_model.zero_grad()
            results = self.inference(curr_model, _curr_batch, curr_training_states, return_elist=True)
            results['loss'].backward(retain_graph=True)
            curr_sz = results['preds'].shape[0]
            total_num_items += curr_sz
            
            for idx, (name, p) in enumerate(curr_model.named_parameters()):
                optpars[idx] = p.data.clone().detach()
                cls_scores[idx] += p.grad.data.clone().pow(2).detach() * curr_sz
            eloss = torch.norm(results['elist'][0]) 
            eloss.backward()
            for idx, (name, p) in enumerate(curr_model.named_parameters()):
                topology_scores[idx] += p.grad.data.clone().pow(2).detach() * curr_sz
        
        for idx, (name, p) in enumerate(curr_model.named_parameters()):
            cls_scores[idx] /= total_num_items
            topology_scores[idx] /= total_num_items
        _idx = curr_training_states['current_task']
        curr_training_states['cls_important_score'][_idx] = cls_scores  
        curr_training_states['topology_important_score'][_idx] = topology_scores
        curr_training_states['optpar'][_idx] = optpars
        curr_training_states['current_task'] += 1
    
class LCClassILTWPTrainer(LCTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        """
            TWP needs three additional parameters `lambda_l`, `lambda_t`, and `beta`.
            `lambda_l` is the hyperparamter for the regularization term (similar to EWC) used in :func:`afterInference`.
            `lambda_t` is the hyperparamter for the regularization term (with topological information) used in :func:`afterInference`.
            `beta` is the hyperparameter for the regularization term related to `cur_importance_score` in :func:`afterInference`.
        """
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.lambda_l = kwargs['lambda_l'] if 'lambda_l' in kwargs else 10000
        self.lambda_t = kwargs['lambda_t']  if 'lambda_t' in kwargs else 10000
        self.beta = kwargs['beta'] if 'beta' in kwargs else 0.1
        
    def inference(self, model, _curr_batch, training_states, return_elist=False):
        """
            The event function to execute inference step.
        
            TWP requires edge weights computed by attention mechanism.
        
            Args:
                model (torch.nn.Module): the current trained model.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
                
            Returns:
                A dictionary containing the inference results, such as prediction result and loss.
        """
        curr_batch, srcs, dsts, labels = _curr_batch
        preds = model(curr_batch.to(self.device), curr_batch.ndata['feat'].to(self.device), srcs, dsts, return_elist=return_elist)
        if return_elist: preds, elist = preds
        loss = self.loss_fn(preds, labels.to(self.device))
        return {'preds': preds, 'loss': loss, 'elist': elist if return_elist else None}
    
    def afterInference(self, results, model, optimizer, _curr_batch, training_states): 
        """
            The event function to execute some processes right after the inference step (for training).
            We recommend performing backpropagation in this event function.
        
            TWP performs regularization process in this function.
        
            Args:
                results (dict): the returned dictionary from the event function `inference`.
                model (torch.nn.Module): the current trained model.
                optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
                
            Returns:
                A dictionary containing the information from the `results`.
        """
        cls_loss = results['loss']
        cls_loss.backward(retain_graph=True)
        cur_importance_score = 0.
        for gs in model.parameters():
            cur_importance_score += torch.norm(gs.grad.data.clone(), p=1) 
        old_importance_score = 0.
        for tt in range(0, training_states['current_task']):
            for i, p in enumerate(model.parameters()):
                I_n = self.lambda_l * training_states['cls_important_score'][tt][i] + self.lambda_t * training_states['topology_important_score'][tt][i]
                I_n = I_n * (p - training_states['optpar'][tt][i]).pow(2)
                old_importance_score += I_n.sum()            
        cls_loss += old_importance_score + self.beta * cur_importance_score
        cls_loss.backward()
        optimizer.step()
        return {'loss': results['loss'].item(), 'acc': self.eval_fn(torch.argmax(results['preds'], dim=-1), _curr_batch[-1].to(self.device))}
    
    def initTrainingStates(self, scenario, model, optimizer):
        return {'current_task':0, 'fisher_loss':{}, 'fisher_att':{}, 'optpar':{}, 'mem_mask':None, 'cls_important_score':{}, 'topology_important_score':{}}
    
    def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):        
        """
            The event function to execute some processes after training the current task.

            TWP computes weights for regularization process and stores the learned weights in this function.
                
            Args:
                task_id (int): the index of the current task.
                curr_dataset (object): The dataset for the current task.
                curr_model (torch.nn.Module): the current trained model.
                curr_optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_training_states (dict): the dictionary containing the current training states.
        """
        curr_model.load_state_dict(curr_training_states['best_weights'])
        optpars = [None for (name, p) in curr_model.named_parameters()]
        cls_scores = [torch.zeros_like(p.data) for (name, p) in curr_model.named_parameters()]
        topology_scores = [torch.zeros_like(p.data) for (name, p) in curr_model.named_parameters()]
        
        train_loader = self.prepareLoader(curr_dataset, curr_training_states)[0]
        total_num_items = 0
        for i, _curr_batch in enumerate(iter(train_loader)):
            curr_model.zero_grad()
            results = self.inference(curr_model, _curr_batch, curr_training_states, return_elist=True)
            results['loss'].backward(retain_graph=True)
            curr_sz = results['preds'].shape[0]
            total_num_items += curr_sz
            
            for idx, (name, p) in enumerate(curr_model.named_parameters()):
                optpars[idx] = p.data.clone().detach()
                cls_scores[idx] += p.grad.data.clone().pow(2).detach() * curr_sz
            eloss = torch.norm(results['elist'][0]) 
            eloss.backward()
            for idx, (name, p) in enumerate(curr_model.named_parameters()):
                topology_scores[idx] += p.grad.data.clone().pow(2).detach() * curr_sz
        
        for idx, (name, p) in enumerate(curr_model.named_parameters()):
            cls_scores[idx] /= total_num_items
            topology_scores[idx] /= total_num_items
        _idx = curr_training_states['current_task']
        curr_training_states['cls_important_score'][_idx] = cls_scores  
        curr_training_states['topology_important_score'][_idx] = topology_scores
        curr_training_states['optpar'][_idx] = optpars
        curr_training_states['current_task'] += 1

class LCTimeILTWPTrainer(LCTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        """
            TWP needs three additional parameters `lambda_l`, `lambda_t`, and `beta`.
            `lambda_l` is the hyperparamter for the regularization term (similar to EWC) used in :func:`afterInference`.
            `lambda_t` is the hyperparamter for the regularization term (with topological information) used in :func:`afterInference`.
            `beta` is the hyperparameter for the regularization term related to `cur_importance_score` in :func:`afterInference`.
        """
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.lambda_l = kwargs['lambda_l'] if 'lambda_l' in kwargs else 10000
        self.lambda_t = kwargs['lambda_t']  if 'lambda_t' in kwargs else 10000
        self.beta = kwargs['beta'] if 'beta' in kwargs else 0.1
        
    def processBeforeTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        """
            The event function to execute some processes before training.
        
            We need to extend the function since the output format is slightly different from the base trainer.
        
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
        
    def inference(self, model, _curr_batch, training_states, return_elist=False):
        """
            The event function to execute inference step.
        
            TWP requires edge weights computed by attention mechanism.
        
            Args:
                model (torch.nn.Module): the current trained model.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
                
            Returns:
                A dictionary containing the inference results, such as prediction result and loss.
        """
        curr_batch, srcs, dsts, labels = _curr_batch
        preds = model(curr_batch.to(self.device), curr_batch.ndata['feat'].to(self.device), srcs, dsts, return_elist=return_elist)
        if return_elist: preds, elist = preds
        loss = self.loss_fn(preds, labels.to(self.device))
        return {'preds': preds, 'loss': loss, 'elist': elist if return_elist else None}
    
    def afterInference(self, results, model, optimizer, _curr_batch, training_states): 
        """
            The event function to execute some processes right after the inference step (for training).
            We recommend performing backpropagation in this event function.
        
            TWP performs regularization process in this function.
        
            Args:
                results (dict): the returned dictionary from the event function `inference`.
                model (torch.nn.Module): the current trained model.
                optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
                
            Returns:
                A dictionary containing the information from the `results`.
        """
        cls_loss = results['loss']
        cls_loss.backward(retain_graph=True)
        cur_importance_score = 0.
        for gs in model.parameters():
            cur_importance_score += torch.norm(gs.grad.data.clone(), p=1) 
        old_importance_score = 0.
        for tt in range(0, training_states['current_task']):
            for i, p in enumerate(model.parameters()):
                I_n = self.lambda_l * training_states['cls_important_score'][tt][i] + self.lambda_t * training_states['topology_important_score'][tt][i]
                I_n = I_n * (p - training_states['optpar'][tt][i]).pow(2)
                old_importance_score += I_n.sum()            
        cls_loss += old_importance_score + self.beta * cur_importance_score
        cls_loss.backward()
        optimizer.step()
        return {'loss': results['loss'].item(), 'acc': self.eval_fn(results['preds'], _curr_batch[-1].to(self.device))}
    
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
        results = self.inference(model, _curr_batch, None)
        return results['preds'], {'loss': results['loss'].item()}
    
    def initTrainingStates(self, scenario, model, optimizer):
        return {'current_task':0, 'fisher_loss':{}, 'fisher_att':{}, 'optpar':{}, 'mem_mask':None, 'cls_important_score':{}, 'topology_important_score':{}}
    
    def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):        
        """
            The event function to execute some processes after training the current task.

            TWP computes weights for regularization process and stores the learned weights in this function.
                
            Args:
                task_id (int): the index of the current task.
                curr_dataset (object): The dataset for the current task.
                curr_model (torch.nn.Module): the current trained model.
                curr_optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_training_states (dict): the dictionary containing the current training states.
        """
        curr_model.load_state_dict(curr_training_states['best_weights'])
        optpars = [None for (name, p) in curr_model.named_parameters()]
        cls_scores = [torch.zeros_like(p.data) for (name, p) in curr_model.named_parameters()]
        topology_scores = [torch.zeros_like(p.data) for (name, p) in curr_model.named_parameters()]
        
        train_loader = self.prepareLoader(curr_dataset, curr_training_states)[0]
        total_num_items = 0
        for i, _curr_batch in enumerate(iter(train_loader)):
            curr_model.zero_grad()
            results = self.inference(curr_model, _curr_batch, curr_training_states, return_elist=True)
            results['loss'].backward(retain_graph=True)
            curr_sz = results['preds'].shape[0]
            total_num_items += curr_sz
            
            for idx, (name, p) in enumerate(curr_model.named_parameters()):
                optpars[idx] = p.data.clone().detach()
                cls_scores[idx] += p.grad.data.clone().pow(2).detach() * curr_sz
            eloss = torch.norm(results['elist'][0]) 
            eloss.backward()
            for idx, (name, p) in enumerate(curr_model.named_parameters()):
                topology_scores[idx] += p.grad.data.clone().pow(2).detach() * curr_sz
        
        for idx, (name, p) in enumerate(curr_model.named_parameters()):
            cls_scores[idx] /= total_num_items
            topology_scores[idx] /= total_num_items
        _idx = curr_training_states['current_task']
        curr_training_states['cls_important_score'][_idx] = cls_scores  
        curr_training_states['topology_important_score'][_idx] = topology_scores
        curr_training_states['optpar'][_idx] = optpars
        curr_training_states['current_task'] += 1

class LPTimeILTWPTrainer(LPTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        """
            TWP needs three additional parameters `lambda_l`, `lambda_t`, and `beta`.
            `lambda_l` is the hyperparamter for the regularization term (similar to EWC) used in :func:`afterInference`.
            `lambda_t` is the hyperparamter for the regularization term (with topological information) used in :func:`afterInference`.
            `beta` is the hyperparameter for the regularization term related to `cur_importance_score` in :func:`afterInference`.
        """
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.lambda_l = kwargs['lambda_l'] if 'lambda_l' in kwargs else 10000
        self.lambda_t = kwargs['lambda_t']  if 'lambda_t' in kwargs else 10000
        self.beta = kwargs['beta'] if 'beta' in kwargs else 0.1
        
    def processTrainIteration(self, model, optimizer, _curr_batch, training_states):
        """
            The event function to handle every training iteration.
        
            TWP performs inference and regularization process in this function.
        
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
        preds = model(graph, feats,
                      srcs.repeat(2), torch.cat((edges[:, 1], neg_dsts), dim=0)).squeeze(-1)
        labels = labels.to(self.device)
        cls_loss = self.loss_fn(preds, torch.cat((labels, torch.zeros_like(labels)), dim=0))
        
        cur_importance_score = 0.
        cls_loss.backward(retain_graph=True)
        for gs in model.parameters():
            cur_importance_score += torch.norm(gs.grad.data.clone(), p=1) 
        old_importance_score = 0.
        for tt in range(0, training_states['current_task']):
            for i, p in enumerate(model.parameters()):
                I_n = self.lambda_l * training_states['cls_important_score'][tt][i] + self.lambda_t * training_states['topology_important_score'][tt][i]
                I_n = I_n * (p - training_states['optpar'][tt][i]).pow(2)
                old_importance_score += I_n.sum()            
        cls_loss += old_importance_score + self.beta * cur_importance_score
        cls_loss.backward()
        optimizer.step()
        del cur_importance_score, old_importance_score
        
        return {'_num_items': preds.shape[0], 'loss': cls_loss.item()}
        
    def initTrainingStates(self, scenario, model, optimizer):
        return {'current_task':0, 'fisher_loss':{}, 'fisher_att':{}, 'optpar':{}, 'mem_mask':None, 'cls_important_score':{}, 'topology_important_score':{}}
    
    def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):        
        """
            The event function to execute some processes after training the current task.

            TWP computes weights for regularization process and stores the learned weights in this function.
                
            Args:
                task_id (int): the index of the current task.
                curr_dataset (object): The dataset for the current task.
                curr_model (torch.nn.Module): the current trained model.
                curr_optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_training_states (dict): the dictionary containing the current training states.
        """
        curr_model.load_state_dict(curr_training_states['best_weights'])
        
        optpars, cls_scores, topology_scores = [], [], []
        train_loader = self.prepareLoader(curr_dataset, curr_training_states)[0]
        total_num_items = 0
        for i, _curr_batch in enumerate(iter(train_loader)):
            curr_model.zero_grad()
            graph, feats = map(lambda x: x.to(self.device), curr_training_states['graph'])
            edges, labels = map(lambda x: x.to(self.device), _curr_batch)
            srcs, dsts = edges[:, 0], edges[:, 1]
            neg_dsts = torch.randint(low=0, high=graph.num_nodes(), size=(srcs.shape[0],)).to(self.device)
            preds, elist = curr_model(graph, feats, srcs.repeat(2), torch.cat((edges[:, 1], neg_dsts), dim=0), return_elist = True)
            preds = preds.squeeze(-1)
            cls_loss = self.loss_fn(preds, torch.cat((labels, torch.zeros_like(labels)), dim=0))
            cls_loss.backward(retain_graph=True)
            
            curr_sz = labels.shape[0]
            total_num_items += curr_sz
            
            if i == 0:
                for idx, (name, p) in enumerate(curr_model.named_parameters()):
                    optpars.append(p.data.clone().detach())
                    cls_scores.append(p.grad.data.clone().pow(2).detach() * curr_sz)
            else:
                for idx, (name, p) in enumerate(curr_model.named_parameters()):
                    cls_scores[idx] += p.grad.data.clone().pow(2).detach() * curr_sz
            
            eloss = torch.norm(elist[0]) 
            eloss.backward()
            
            if i == 0:
                for idx, (name, p) in enumerate(curr_model.named_parameters()):
                    topology_scores.append(p.grad.data.clone().pow(2).detach() * curr_sz)
            else:
                for idx, (name, p) in enumerate(curr_model.named_parameters()):
                    topology_scores[idx] += p.grad.data.clone().pow(2).detach() * curr_sz

            len_elist = len(elist)
            for _ in range(len_elist):
                del elist[0]
            del cls_loss, eloss, elist
        
        for idx, (name, p) in enumerate(curr_model.named_parameters()):
            cls_scores[idx] /= total_num_items
            topology_scores[idx] /= total_num_items
            
        _idx = curr_training_states['current_task']
        curr_training_states['cls_important_score'][_idx] = cls_scores  
        curr_training_states['topology_important_score'][_idx] = topology_scores
        curr_training_states['optpar'][_idx] = optpars
        curr_training_states['current_task'] += 1         
        
class LPDomainILTWPTrainer(LPTimeILTWPTrainer):
    """
        This trainer has the same behavior as `LPTimeILTWPTrainer`.
    """
    pass