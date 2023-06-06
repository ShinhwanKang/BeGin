import sys
import numpy as np
import torch
import copy
import re
import dgl
from torch import nn
import torch.nn.functional as F
from begin.trainers.graphs import GCTrainer

class GCTaskILHATTrainer(GCTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.smax = kwargs['smax']
        self.lamb = kwargs['lamb']
        
        self.clipgrad = kwargs.get('clipgrad', 10000.)
        self.thres_cosh = kwargs.get('thres_cosh', 50.)
        self.thres_emb = kwargs.get('thres_emb', 6.)
        
        self.mask_pre = None
        self.mask_back = None
        
        self.embedding_list = []
        self.embedding_list.append(nn.Embedding(self.num_tasks, model.n_hidden))
        for i in range(model.n_layers):
            self.embedding_list.append(nn.Embedding(self.num_tasks, model.n_hidden))
        for i in range(model.n_mlp_layers):
            self.embedding_list.append(nn.Embedding(self.num_tasks, model.n_hidden // (1 << (i+1))))
        
        self.masking = []
        for i in range(1 + model.n_layers + model.n_mlp_layers):
            self.masking.append(None)
        
        self.gate = nn.Sigmoid()
        
    def initTrainingStates(self, scenario, model, optimizer):
        return {'class_to_task': -torch.ones(model.classifier.num_outputs, dtype=torch.long)}
    
    def criterion(self, outputs, targets, masks):
        reg=0
        count=0
        if self.mask_pre is not None:
            for m,mp in zip(masks,self.mask_pre):
                aux=1-mp
                reg+=(m*aux).sum()
                count+=aux.sum()
        else:
            for m in masks:
                reg+=m.sum()
                count+=np.prod(m.size()).item()
        reg/=count
        return self.loss_fn(outputs,targets)+self.lamb*reg

    def get_mask(self, model, task_id, s, m, e):
        for i in range(1 + model.n_layers + model.n_mlp_layers):
            m[i]=(self.gate(s*e[i](torch.tensor(task_id))))
        return m
    
    def processBeforeTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        """
            The event function to execute some processes before training.
            
            Args:
                task_id (int): the index of the current task
                curr_dataset (object): The dataset for the current task.
                curr_model (torch.nn.Module): the current trained model.
                curr_optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_training_states (dict): the dictionary containing the current training states.
        """
        super().processBeforeTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)
        
        # detect new classes
        if self.curr_task == 0:
            curr_model.class_to_task = curr_training_states.pop('class_to_task')    
        trainloader, valloader, _ = self.prepareLoader(curr_dataset, curr_training_states)
        first_train_batch = next(iter(trainloader))
        curr_model.class_to_task[first_train_batch[-1][0]] = self.curr_task
        
        # variable for mini-batch training
        curr_training_states['num_batches'] = len(trainloader)
        curr_training_states['curr_cnt'] = 0
        
    def inference(self, model, _curr_batch, training_states):
        """
            The event function to execute inference step.
        
            For task-IL, we need to additionally consider task information for the inference step.
            In addition, we need to consider (attentive) masks for HAT.
            
            Args:
                model (torch.nn.Module): the current trained model.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
                
            Returns:
                A dictionary containing the inference results, such as prediction result and loss.
        """
        if ('curr_cnt' not in training_states) or ('num_batches' not in training_states):
            curr_s = self.smax
        else:
            curr_s = (1./self.smax) + ((self.smax-(1./self.smax)) * (training_states['curr_cnt'] / (training_states['num_batches'] - 1)))
        hat_masks = self.get_mask(model, training_states['current_task'], curr_s, self.masking, self.embedding_list)
        
        graphs, labels, masks = _curr_batch
        preds = model(graphs.to(self.device),
                      graphs.ndata['feat'].to(self.device) if 'feat' in graphs.ndata else None,
                      edge_attr = graphs.edata['feat'].to(self.device) if 'feat' in graphs.edata else None,
                      edge_weight = graphs.edata['weight'].to(self.device) if 'weight' in graphs.edata else None,
                      task_masks = masks)
        loss = self.loss_fn(preds, labels.to(self.device))
        return {'preds': preds, 'loss': loss, 'hat_masks': hat_masks, 'curr_s': curr_s}
    
    def afterInference(self, results, model, optimizer, _curr_batch, training_states):
        """
            The event function to execute some processes right after the inference step (for training).
            We recommend performing backpropagation in this event function.
            
            For HAT, we need to consider the techniques introduced in the original paper in this event function.
            
            Args:
                results (dict): the returned dictionary from the event function `inference`.
                model (torch.nn.Module): the current trained model.
                optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
                use_mask (bool): whether model masks weights of the model.
                
            Returns:
                A dictionary containing the information from the `results`.
        """
        # Backward
        graphs, labels, masks = _curr_batch
        results['loss'] = self.criterion(results['preds'], labels.to(self.device), results['hat_masks'])
        results['loss'].backward()
        
        # Restrict layer gradients in backprop
        if self.curr_task>0:
            for n, p in model.named_parameters():
                if n in self.mask_back:
                    p.grad.data*=self.mask_back[n].to(self.device)
                    
        # Compensate embedding gradients
        for p in self.embedding_list:
            num=torch.cosh(torch.clamp(results['curr_s']*p.weight.data,-self.thres_cosh,self.thres_cosh))+1.
            den=torch.cosh(p.weight.data)+1.
            p.weight.grad.data*=(self.smax*num)/(results['curr_s']*den)
            
        # Apply step
        torch.nn.utils.clip_grad_norm_(model.parameters(),self.clipgrad)            
        optimizer.step()
        training_states['curr_cnt'] += 1
        
        return {'_num_items': results['preds'].shape[0], 'loss': results['loss'].item(), 'acc': self.eval_fn(results['preds'].argmax(-1), labels.to(self.device))}
    
    def get_view_for(self, name, curr_model, parameter, masks):
        if 'enc' in name:
            target_num = 0
        else:
            target_num = list(map(int, re.findall('\d+', name)))[0] + 1
            if 'mlp_layers' in name:
                target_num = target_num + curr_model.n_layers
        if 'bias' in name:
                return masks[target_num].data.view(-1)
        else:
            if 'enc' in name:
                return masks[0].data.unsqueeze(-1).expand_as(parameter.data)
            elif 'mlp_layers' in name:
                post = masks[target_num].data.unsqueeze(-1).expand_as(parameter.data)
                pre = masks[target_num-1].data.unsqueeze(0).expand_as(parameter.data)
                return torch.min(post,pre)
            else:
                post = masks[target_num].data.unsqueeze(0).expand_as(parameter.data)
                pre = masks[target_num-1].data.unsqueeze(-1).expand_as(parameter.data)
                return torch.min(post,pre)
            
    def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        """
            The event function to execute some processes after training the current task.
            
            In this event function, our implementation updates the masks for running HAT.
            
            Args:
                task_id (int): the index of the current task.
                curr_dataset (object): The dataset for the current task.
                curr_model (torch.nn.Module): the current trained model.
                curr_optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_training_states (dict): the dictionary containing the current training states.
        """
        
        super().processAfterTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)
        
        # Activations mask
        mask = self.get_mask(curr_model, task_id, self.smax, self.masking, self.embedding_list)
        for i in range(len(mask)):
            mask[i]=torch.autograd.Variable(mask[i].data.clone(),requires_grad=False)
        if task_id==0:
            self.mask_pre=mask
        else:
            for i in range(len(self.mask_pre)):
                self.mask_pre[i]=torch.max(self.mask_pre[i],mask[i])
                
        # Weights mask
        self.mask_back={}
        for name, param in curr_model.named_parameters():
            if 'enc' in name or 'convs' in name or 'mlp_layers' in name:
                vals=self.get_view_for(name, curr_model, param, self.mask_pre)
                if vals is not None:
                    self.mask_back[name]=1-vals
                    
    def processTrainIteration(self, model, optimizer, _curr_batch, training_states):
        """
            The event function to handle every training iteration.
        
            Args:
                model (torch.nn.Module): the current trained model.
                optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
                use_mask (bool): whether model masks weights of the model.
                
            Returns:
                A dictionary containing the outcomes (stats) during the training iteration.
        """
        training_states['current_task'] = self.curr_task
        
        optimizer.zero_grad()
        self.beforeInference(model, optimizer, _curr_batch, training_states)
        inference_results = self.inference(model, _curr_batch, training_states)
        return self.afterInference(inference_results, model, optimizer, _curr_batch, training_states)
    
    def processEvalIteration(self, model, _curr_batch):
        """
            The event function to handle every evaluation iteration.
            
            We need to extend the base function since the output format is slightly different from the base trainer.
            
            Args:
                model (torch.nn.Module): the current trained model.
                curr_batch (object): the data (or minibatch) for the current iteration.
                use_mask (bool): whether model masks weights of the model.
                
            Returns:
                A dictionary containing the outcomes (stats) during the evaluation iteration.
        """
        
        # classify each sample by its task-id
        eval_model = copy.deepcopy(model)
        graphs, labels, mask = _curr_batch
        each_graph = dgl.unbatch(graphs)
        task_ids = torch.max(mask * model.class_to_task, dim=-1).values
        task_checks = torch.min(mask * model.class_to_task, dim=-1).values
        task_ids[task_checks < 0] = self.curr_task
        
        num_samples = torch.bincount(task_ids.detach().cpu(), minlength=self.curr_task+1)
        total_results = torch.zeros(mask.shape[0], dtype=torch.long).to(self.device)
        total_loss = 0.
        
        # handle the samples with different tasks separately
        for i in range(self.curr_task, -1, -1):
            if num_samples[i].item() == 0: continue
            eval_mask = (task_ids == i)
            eval_nodes = torch.nonzero(eval_mask, as_tuple=True)[0]
            target_graphs = dgl.batch([each_graph[graph_id] for graph_id in eval_nodes.tolist()])
            results = self.inference(eval_model, (target_graphs, labels[eval_mask], mask[eval_mask]), {'current_task': i})
            
            total_results[eval_nodes] = torch.argmax(results['preds'], dim=-1)
            total_loss += results['loss'].item() * num_samples[i].item()
        n_samples = torch.sum(num_samples).item()
        total_loss /= n_samples
        
        return total_results, {'loss': total_loss, 'n_samples': n_samples}