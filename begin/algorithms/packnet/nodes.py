import sys
import numpy as np
import torch
import copy
from torch import nn
import torch.nn.functional as F
from begin.trainers.nodes import NCTrainer

class NCTaskILPackNetTrainer(NCTrainer):
    def initTrainingStates(self, scenario, model, optimizer):
        model_masks = {}
        with torch.no_grad():
            pr = 1. - np.exp((1. / (self.num_tasks - 1)) * np.log(1. / self.num_tasks))
            for name, p in model.named_parameters():
                if 'convs' in name:
                    model_masks[name] = (torch.ones_like(p.data) * (self.num_tasks + 1)).long()
        return {'pr': pr, 'packnet_masks': model_masks, 'class_to_task': -torch.ones(model.classifier.num_outputs, dtype=torch.long)}
    
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
    
    def processBeforeTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        """
            The event function to execute some processes before training.
            
            PackNet masks the parameters of the model depending on the current magnitude of parameters in this function.
                   
            Args:
                task_id (int): the index of the current task
                curr_dataset (object): The dataset for the current task.
                curr_model (torch.nn.Module): the current trained model.
                curr_optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_training_states (dict): the dictionary containing the current training states.
        """
        
        # fix batchnorm parameters
        def set_bn_eval(m):
            if isinstance(m, nn.BatchNorm1d):
                m.eval()
        super().processBeforeTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)
        
        if self.curr_task == 0:
            # initialize masks
            curr_model.packnet_masks = curr_training_states.pop('packnet_masks')
            curr_model.class_to_task = curr_training_states.pop('class_to_task')
        else:
            # fix batchnorm parameters
            for name, p in curr_model.named_parameters():
                if 'norms' in name:
                    p.requires_grad_(False)
                    
        # detect new classes    
        new_classes = curr_dataset.ndata['task_specific_mask'][curr_dataset.ndata['train_mask']][0]
        curr_model.class_to_task[new_classes > 0] = self.curr_task
        
        # pre-training (10% of number of training epochs)
        trainset, valset, _ = self.prepareLoader(curr_dataset, curr_training_states)
        pre_scheduler = self.scheduler_fn(curr_optimizer)
        best_val_loss = 1e10
        pre_checkpoint = copy.deepcopy(curr_model.state_dict())
        for epoch_cnt in range(self.max_num_epochs // 10):
            curr_model.train()
            if self.curr_task > 0:
                curr_model.apply(set_bn_eval)
            train_dict = self.processTrainIteration(curr_model, curr_optimizer, trainset[0], curr_training_states, use_mask=False)
            curr_model.eval()
            _, val_dict = self.processEvalIteration(curr_model, valset[0], use_mask=False)
            val_loss = val_dict['loss']

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                pre_checkpoint = copy.deepcopy(curr_model.state_dict())
            pre_scheduler.step(val_loss)
            if -1e-9 < (curr_optimizer.param_groups[0]['lr'] - pre_scheduler.min_lrs[0]) < 1e-9:
                break
                
        # select parameters and perform masking
        with torch.no_grad():
            curr_model.load_state_dict(pre_checkpoint)
            for name, p in curr_model.named_parameters():
                if 'convs' in name:
                    try:
                        candidates = torch.abs(p.data)[curr_model.packnet_masks[name] >= self.curr_task]
                        threshold = torch.topk(candidates, int((candidates.shape[0] * curr_training_states['pr']) + 0.5), largest=True).values.min()
                        accept_criteria = (curr_model.packnet_masks[name] >= self.curr_task)
                        if self.curr_task < self.num_tasks - 1:
                            accept_criteria &= (torch.abs(p.data) >= threshold)
                        curr_model.packnet_masks[name][accept_criteria] = self.curr_task
                        p.data[curr_model.packnet_masks[name] > self.curr_task] = 0.
                    except:
                        pass
                    
        for name, p in curr_model.named_parameters():
            if 'norms' in name:
                p.requires_grad_(False)
        
        super().processBeforeTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)
        
    def afterInference(self, results, model, optimizer, _curr_batch, training_states, use_mask=True):
        """
            The event function to execute some processes right after the inference step (for training).
            We recommend performing backpropagation in this event function.
            
            In this function, PackNet weights gradients of parametrs according to 'packnet_masks'.
            For this, Packnet additionally needs 'use_mask' parameter.
            
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
        results['loss'].backward()
        if use_mask:
            # flow gradients only for masked ones
            for name, p in model.named_parameters():
                if 'convs' in name:
                    p.grad = p.grad * (model.packnet_masks[name] == self.curr_task).long()
        else:
            # flow gradients only for unmasked ones (pre-training)
            for name, p in model.named_parameters():
                if 'convs' in name:
                    p.grad = p.grad * (model.packnet_masks[name] >= self.curr_task).long()
        optimizer.step()
        return {'loss': results['loss'].item(), 'acc': self.eval_fn(results['preds'].argmax(-1), _curr_batch[0].ndata['label'][_curr_batch[1]].to(self.device))}
    
    def processTrainIteration(self, model, optimizer, _curr_batch, training_states, use_mask=True):
        """
            The event function to handle every training iteration.
        
            PackNet additionally needs 'use_mask' parameter.
        
            Args:
                model (torch.nn.Module): the current trained model.
                optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
                use_mask (bool): whether model masks weights of the model.
                
            Returns:
                A dictionary containing the outcomes (stats) during the training iteration.
        """
        
        # freeze running variables of batchnorms
        if use_mask:
            def set_bn_eval(m):
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()
            model.apply(set_bn_eval)
        
        optimizer.zero_grad()
        self.beforeInference(model, optimizer, _curr_batch, training_states)
        inference_results = self.inference(model, _curr_batch, training_states)
        return self.afterInference(inference_results, model, optimizer, _curr_batch, training_states, use_mask=use_mask)
    
    def processEvalIteration(self, model, _curr_batch, use_mask=True):
        """
            The event function to handle every evaluation iteration.
            
            We need to extend the base function since the output format is slightly different from the base trainer.
            
            PackNet additionally needs 'use_mask' parameter.
            
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
            if use_mask:
                for name, p in eval_model.named_parameters():
                    if 'convs' in name:
                        p.data = p.data * (model.packnet_masks[name] <= i).long()

            eval_mask = torch.zeros(mask.shape[0])
            eval_mask[test_nodes[task_ids == i]] = 1
            results = self.inference(eval_model, (curr_batch, eval_mask.bool()), None)

            total_results[task_ids == i] = torch.argmax(results['preds'], dim=-1)
            total_loss += results['loss'].item() * num_samples[i].item()
        total_loss /= torch.sum(num_samples).item()

        return total_results, {'loss': total_loss}