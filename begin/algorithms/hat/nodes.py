import sys
import numpy as np
import torch
import copy
import re
from torch import nn
import torch.nn.functional as F
from begin.trainers.nodes import NCTrainer

class NCTaskILHATTrainer(NCTrainer):
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
        for i in range(model.n_layers):
            self.embedding_list.append(nn.Embedding(self.num_tasks, model.n_hidden))
        
        self.masking = []
        for i in range(model.n_layers):
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
    
    def get_mask(self, task_id, model, s, m, e):
        for i in range(model.n_layers):
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
        new_classes = curr_dataset.ndata['task_specific_mask'][curr_dataset.ndata['train_mask']][0]
        curr_model.class_to_task[new_classes > 0] = self.curr_task
        curr_training_states['scheduler'] = self.scheduler_fn(curr_optimizer)
    
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
        hat_masks = self.get_mask(training_states['current_task'], model, self.smax, self.masking, self.embedding_list)
        curr_batch, mask = _curr_batch
        preds = model.forward_hat(curr_batch.to(self.device), curr_batch.ndata['feat'].to(self.device), hat_masks=hat_masks, task_masks=curr_batch.ndata['task_specific_mask'].to(self.device))[mask]
        loss = self.loss_fn(preds, curr_batch.ndata['label'][mask].to(self.device))
        return {'preds': preds, 'loss': loss, 'hat_masks': hat_masks}
    
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
        curr_batch, mask = _curr_batch
        results['loss'] = self.criterion(results['preds'], curr_batch.ndata['label'][mask].to(self.device), results['hat_masks'])
        results['loss'].backward()
        
        # Restrict layer gradients in backprop
        if self.curr_task>0:
            for n, p in model.named_parameters():
                if n in self.mask_back:
                    p.grad.data*=self.mask_back[n].to(self.device)
        
        # Compensate embedding gradients
        for p in self.embedding_list:
            num=torch.cosh(torch.clamp(self.smax*p.weight.data,-self.thres_cosh,self.thres_cosh))+1
            den=torch.cosh(p.weight.data)+1
            p.weight.grad.data*=num/den
        
        # Apply step
        torch.nn.utils.clip_grad_norm_(model.parameters(),self.clipgrad)                
        optimizer.step()
        return {'loss': results['loss'].item(), 'acc': self.eval_fn(results['preds'].argmax(-1), _curr_batch[0].ndata['label'][_curr_batch[1]].to(self.device))}
    
    def get_view_for(self, name, parameter, masks):
        target_num = list(map(int, re.findall('\d+', name)))[0]
        if target_num==0:
            return masks[0].data.expand_as(parameter.data)
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
        mask = self.get_mask(task_id, curr_model, self.smax, self.masking, self.embedding_list)

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
            if "convs" not in name:
                continue
            vals=self.get_view_for(name, param, self.mask_pre)
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
        training_states['current_task']=self.curr_task
        
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
        curr_batch, mask = _curr_batch
        task_ids = torch.max(curr_batch.ndata['task_specific_mask'][mask] * model.class_to_task, dim=-1).values
        task_checks = torch.min(curr_batch.ndata['task_specific_mask'][mask] * model.class_to_task, dim=-1).values
        test_nodes = torch.arange(mask.shape[0])[mask]
        task_ids[task_checks < 0] = self.curr_task

        num_samples = torch.bincount(task_ids.detach().cpu())
        total_results = torch.zeros_like(test_nodes).to(self.device)
        total_loss = 0.
        
        # handle the samples with different tasks separately
        for i in range(self.curr_task, -1, -1):
            if num_samples[i].item() == 0: continue

            eval_mask = torch.zeros(mask.shape[0])
            eval_mask[test_nodes[task_ids == i]] = 1
            
            states = {'current_task':i}
            results = self.inference(eval_model, (curr_batch, eval_mask.bool()), states)

            total_results[task_ids == i] = torch.argmax(results['preds'], dim=-1)
            total_loss += results['loss'].item() * num_samples[i].item()
        total_loss /= torch.sum(num_samples).item()

        return total_results, {'loss': total_loss}