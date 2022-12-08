import sys
import numpy as np
import torch
import copy
import torch.nn.functional as F
from begin.trainers.nodes import NCTrainer
from .utils import *

from torch.nn.parameter import Parameter

class NCTaskILPIGGYBACKTrainer(NCTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        """
            `threshold_fn` is a function converting a real-valued mask to a binary mask.
            `masked_weights` is used for storing weights of the backbone network.
        """
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        
        self.threshold_fn_name = kwargs['threshold_fn']
        self.threshold = kwargs['threshold']
        
        # Initialize real-valued mask weights.
        self.binary_masks = []
        
        # Initialize the thresholder.
        if self.threshold_fn_name == 'binarizer':
            # Calling binarizer with threshold
            self.threshold_fn = Binarizer(threshold=self.threshold)
        elif self.threshold_fn_name == 'ternarizer':
            # Calling ternarizer with threshold
            self.threshold_fn = Ternarizer(threshold=self.threshold)
        
        self.masked_weights={}
        
    def initTrainingStates(self, scenario, model, optimizer):
        return {'class_to_task': -torch.ones(model.classifier.num_outputs, dtype=torch.long)}
    

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
        if self.curr_task == 0:
            curr_model.class_to_task = curr_training_states.pop('class_to_task')
        new_classes = curr_dataset.ndata['task_specific_mask'][curr_dataset.ndata['train_mask']][0]
        curr_model.class_to_task[new_classes > 0] = self.curr_task
        
        if (len(self.binary_masks)==task_id):
            # initialize binary_masks of each task
            masks={}
            for name, param in curr_model.named_parameters():
                if 'conv' not in name:
                    continue
                masks[name]=Parameter(torch.randn(param.size()), requires_grad=True)
            self.binary_masks.append(masks)
            
    def maskedModel(self, model, task_id=None):
        for name, param in model.named_parameters():
            if 'conv' not in name:
                continue
            self.masked_weights[name]=param.data
            
            # Get binarized/ternarized mask from real-valued mask.
            if task_id == None:
                mask_thresholded = self.threshold_fn.apply(self.binary_masks[-1][name], self.threshold)
            else:
                mask_thresholded = self.threshold_fn.apply(self.binary_masks[task_id][name], self.threshold)
            # Mask weights with above mask.
            weights = param.data
            device = weights.get_device()
            weight_thresholded = mask_thresholded.to(device) * weights
            param.data = weight_thresholded
        
        return model
    
    def unmaskedModel(self, model):
        for name, param in model.named_parameters():
            if 'conv' not in name:
                continue
            param.data = self.masked_weights[name]
        return model
    
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
        model = self.maskedModel(model)
        
    
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
            
            Piggyback un-mask weights of masked parameters of the model.
            
            Args:
                results (dict): the returned dictionary from the event function `inference`.
                model (torch.nn.Module): the current trained model.
                optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
                
            Returns:
                A dictionary containing the information from the `results`.
        """
        results['loss'].backward()
        optimizer.step()
        model = self.unmaskedModel(model)
        return {'loss': results['loss'].item(), 'acc': self.eval_fn(results['preds'].argmax(-1), _curr_batch[0].ndata['label'][_curr_batch[1]].to(self.device))}
    

    def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        """
            The event function to execute some processes after training the current task.
            
            After a task, Piggyback un-mask weights of masked parameters of the model.
            
            Args:
                task_id (int): the index of the current task.
                curr_dataset (object): The dataset for the current task.
                curr_model (torch.nn.Module): the current trained model.
                curr_optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_training_states (dict): the dictionary containing the current training states.
        """
        super().processAfterTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)
        curr_model = self.unmaskedModel(curr_model)
    
    def processEvalIteration(self, model, _curr_batch, task_id = None):
        """
            The event function to handle every evaluation iteration.
            
            Piggyback has to use different models (weights) to evaluate the performance for each task.
            
            Args:
                model (torch.nn.Module): the current trained model.
                curr_batch (object): the data (or minibatch) for the current iteration.
                task_id (int): the id of a task
                
            Returns:
                A dictionary containing the outcomes (stats) during the evaluation iteration.
        """
        eval_model = copy.deepcopy(model)
        curr_batch, mask = _curr_batch
        task_ids = torch.max(curr_batch.ndata['task_specific_mask'][mask] * eval_model.class_to_task, dim=-1).values
        test_nodes = torch.arange(mask.shape[0])[mask]
        
        task_ids[task_ids < 0] = self.curr_task
        
        num_samples = torch.bincount(task_ids.detach().cpu())
        
        total_results = torch.zeros_like(test_nodes).to(self.device)
        total_loss = 0.
        for i in range(self.curr_task, -1, -1):
            if num_samples[i].item() == 0: continue
            if task_id != None:
                eval_model = self.maskedModel(eval_model, task_id)
            
            eval_mask = torch.zeros(mask.shape[0])
            eval_mask[test_nodes[task_ids == i]] = 1
            results = self.inference(eval_model, (curr_batch, eval_mask.bool()), None)    
            
            if task_id != None:
                eval_model = self.unmaskedModel(eval_model)
            
            total_results[task_ids == i] = torch.argmax(results['preds'], dim=-1)
            total_loss += results['loss'].item() * num_samples[i].item()
        total_loss /= torch.sum(num_samples).item()
             
        del eval_model
        
        return total_results, {'loss': total_loss}
    