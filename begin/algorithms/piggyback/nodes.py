import sys
import numpy as np
import torch
import copy
import torch.nn.functional as F
from begin.trainers.nodes import NCTrainer

from .utils import *
from torch import nn

class NCTaskILPiggybackTrainer(NCTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        """
            `threshold_fn` is a function converting a real-valued mask to a binary mask.
            `masked_weights` is used for storing weights of the backbone network.
        """
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.threshold = kwargs['threshold']
        
    def initTrainingStates(self, scenario, model, optimizer):
        return {'class_to_task': -torch.ones(model.classifier.num_outputs, dtype=torch.long)}
    
    def beforeInference(self, model, optimizer, _curr_batch, training_states):
        """
            The event function to execute some processes right before inference (for training).
            
            Before training, Piggyback gets binarized/ternarized mask from real-valued mask and weights the masks.
            
            Args:
                model (torch.nn.Module): the current trained model.
                optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
        """
        # fix batchnorm parameters
        def set_bn_eval(m):
            if isinstance(m, nn.BatchNorm1d):
                m.eval()
        model.apply(set_bn_eval)
        for name, p in model.named_parameters():
            if 'norm' in name:
                p.requires_grad_(False)
                
        # masking paramters
        ccnt = 0
        weights_before_inference = []
        for name, p in model.named_parameters():
            if 'conv' in name:
                weights_before_inference.append(copy.deepcopy(p))
                p.data.copy_(weights_before_inference[-1] * (model.task_masks[ccnt] >= self.threshold))
                ccnt += 1
        return weights_before_inference
    
        
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
            
    def afterInference(self, results, model, optimizer, _curr_batch, training_states):
        """
            The event function to execute some processes right after the inference step (for training).
            We recommend performing backpropagation in this event function.
            In our implementation, we compute the gradient for the real-valued masks and restore the masked parameters.
            
            Args:
                results (dict): the returned dictionary from the event function `inference`.
                model (torch.nn.Module): the current trained model.
                optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
                
            Returns:
                A dictionary containing the information from the `results`.
        """
        # compute gradient for masks
        results['loss'].backward()
        ccnt = 0
        for name, p in model.named_parameters():
            if 'conv' in name:
                model.task_masks[ccnt].grad = results['_before_inference'][ccnt].detach() * p.grad
                ccnt += 1
        optimizer.step()
        training_states['mask_optimizer'].step()
        
        # restore the masked parameters
        ccnt = 0
        with torch.no_grad():
            for name, p in model.named_parameters():
                if 'conv' in name:
                    p.data.copy_(results['_before_inference'][ccnt])
                    ccnt += 1
        return {'loss': results['loss'].item(), 'acc': self.eval_fn(results['preds'].argmax(-1), _curr_batch[0].ndata['label'][_curr_batch[1]].to(self.device))}
    
    def processBeforeTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        """
            The event function to execute some processes before training.
            In this function, masks for network parameters are initialized.
            
            Args:
                task_id (int): the index of the current task
                curr_dataset (object): The dataset for the current task.
                curr_model (torch.nn.Module): the current trained model.
                curr_optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_training_states (dict): the dictionary containing the current training states.
        """
        if self.curr_task == 0:
            # pre-training with Deep Graph Infomax
            trainloader, _1, _2 = self.prepareLoader(curr_dataset, curr_training_states)
            pre_model = copy.deepcopy(curr_model)
            dgi_model = DGI(pre_model).to(self.device)
            pre_optimizer = self.optimizer_fn(dgi_model.parameters())
            pre_scheduler = self.scheduler_fn(pre_optimizer)
            best_val_loss = 1e10
            for epoch_cnt in range(self.max_num_epochs):
                val_loss = 0.
                for _curr_batch in trainloader:
                    pre_optimizer.zero_grad()
                    curr_batch, _ = _curr_batch
                    _loss = dgi_model(curr_batch.to(self.device), curr_batch.ndata['feat'].to(self.device))
                    _loss.backward()
                    pre_optimizer.step()    
                    val_loss = val_loss + _loss.item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    pre_checkpoint = copy.deepcopy(dgi_model.encoder.state_dict())
                pre_scheduler.step(val_loss)
                if -1e-9 < (pre_optimizer.param_groups[0]['lr'] - pre_scheduler.min_lrs[0]) < 1e-9:
                    break
            curr_model.load_state_dict(pre_checkpoint)
        
        super().processBeforeTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)
        
        # initialize masks and fix batchnorm
        _task_masks = []
        for name, p in curr_model.named_parameters():
            if 'conv' in name:
                _task_masks.append(nn.Parameter(torch.ones_like(p.data) * self.threshold * 2.))
            if 'norm' in name:
                p.requires_grad_(False)
        curr_model.task_masks = _task_masks
        
        # define optimizer for masks
        curr_training_states['mask_optimizer'] = self.optimizer_fn(curr_model.task_masks)
        
        # detect new classes
        if self.curr_task == 0:
            curr_model.class_to_task = curr_training_states.pop('class_to_task')
            curr_model.done_masks = []    
        new_classes = curr_dataset.ndata['task_specific_mask'][curr_dataset.ndata['train_mask']][0]
        curr_model.class_to_task[new_classes > 0] = self.curr_task
        curr_training_states['scheduler'] = self.scheduler_fn(curr_optimizer)
        
    def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        """
            The event function to execute some processes after training the current task.
            
            In this event function, our implementation stores the learned masks for each task.
            
            Args:
                task_id (int): the index of the current task.
                curr_dataset (object): The dataset for the current task.
                curr_model (torch.nn.Module): the current trained model.
                curr_optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_training_states (dict): the dictionary containing the current training states.
        """
        
        super().processAfterTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)
        curr_model.done_masks.append(copy.deepcopy(curr_model.task_masks))
    
    def processAfterEachIteration(self, curr_model, curr_optimizer, curr_training_states, curr_iter_results):
        """
            The event function to execute some processes for every end of each epoch.
            Whether to continue training or not is determined by the return value of this function.
            If the returned value is False, the trainer stops training the current model in the current task.
            
            In this event function, our implementation synchronize the learning rates of the main optimizer and the optimizer for masks.
            
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
        ret = super().processAfterEachIteration(curr_model, curr_optimizer, curr_training_states, curr_iter_results)
        # sync learning rate
        curr_training_states['mask_optimizer'].param_groups[0]['lr'] = curr_optimizer.param_groups[0]['lr']
        return ret
    
    def processEvalIteration(self, model, _curr_batch):
        """
            The event function to handle every evaluation iteration.
            
            Piggyback has to use different masks to evaluate the performance for each task.
            
            Args:
                model (torch.nn.Module): the current trained model.
                curr_batch (object): the data (or minibatch) for the current iteration.
                task_id (int): the id of a task
                
            Returns:
                A dictionary containing the outcomes (stats) during the evaluation iteration.
        """
        
        # classify each sample by its task-id
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
            eval_model = copy.deepcopy(model)
            if i < self.curr_task:
                eval_model.task_masks = eval_model.done_masks[i]
            if num_samples[i].item() == 0: continue
            eval_mask = torch.zeros(mask.shape[0])
            eval_mask[test_nodes[task_ids == i]] = 1
            
            before_inference_results = self.beforeInference(eval_model, None, _curr_batch, None)
            results = self.inference(eval_model, (curr_batch, eval_mask.bool()), None)
            ccnt = 0
            with torch.no_grad():
                for name, p in eval_model.named_parameters():
                    if 'conv' in name:
                        p.data.copy_(before_inference_results[ccnt])
                        ccnt += 1
        
            total_results[task_ids == i] = torch.argmax(results['preds'], dim=-1)
            total_loss += results['loss'].item() * num_samples[i].item()
        total_loss /= torch.sum(num_samples).item()

        return total_results, {'loss': total_loss}