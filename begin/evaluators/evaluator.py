import torch
from torch import nn
from torch_scatter import scatter
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import time

class BaseEvaluator:
    r"""
        Base class for evaluating the performance.
        Users can create their own evaluator by extending this class.
        
        Arguments:
            num_tasks (int): The number of tasks in the target scenario.
            task_ids (torch.Tensor): task ids of each instance.
    """
    
    def __init__(self, num_tasks, task_ids):
        self.num_tasks = num_tasks
        self._task_ids = task_ids

    def __call__(self, prediction, answer, indices):
        r"""
            Measure the performance on each task.
        
            Args:
                prediction (torch.Tensor): predicted output of the current model
                answer (torch.Tensor): ground-truth answer
                indices (torch.Tensor): indexes of the chosen instances for evaluation
        """
        raise NotImplementedError
        
    def simple_eval(self, prediction, answer):
        r"""
            Compute performance for the given batch when we ignore task configuration.
            During the training procedure, this function is called by the function get_simple_eval_result implemented in ScenarioLoaders.
            
            Args:
                prediction (torch.Tensor): predicted output of the current model
                answer (torch.Tensor): ground-truth answer
        """
        raise NotImplementedError
        
class AccuracyEvaluator(BaseEvaluator):
    r"""
        The evaluator for computing accuracy.
        
        Bases: ``BaseEvaluator``
    """
    def __call__(self, _prediction, _answer, indices):
        prediction = _prediction.squeeze().to(_answer.device)
        answer = _answer.squeeze()
        scope = self._task_ids[indices] < self.num_tasks
        accuracy_per_task = scatter((prediction == answer).float(), self._task_ids[indices], dim=-1, reduce='mean', dim_size = self.num_tasks + 1)
        accuracy_per_task[self.num_tasks] = self.simple_eval(prediction[scope], answer[scope])
        return accuracy_per_task
    
    def simple_eval(self, prediction, answer):
        return ((prediction.squeeze().to(answer.device) == answer.squeeze()).float().sum() / answer.shape[0]).item()

class ROCAUCEvaluator(BaseEvaluator):
    r"""
        The evaluator for computing ROCAUC score.
        
        Bases: ``BaseEvaluator``
    """
    def __call__(self, _prediction, answer, indices):
        prediction = _prediction.to(answer.device)
        target_ids = self._task_ids[indices]
        retval = torch.zeros(self.num_tasks + 1)
        for i in range(self.num_tasks):
            is_target = (target_ids == i)
            if is_target.any(): retval[i] = self.simple_eval(prediction[is_target], answer[is_target])
        retval[self.num_tasks] = self.simple_eval(prediction, answer)

        return retval
    
    def simple_eval(self, prediction, answer):
        num_items, num_qs = answer.shape
        valid_cols = (answer.sum(0) < num_items) & (answer.sum(0) >= 0)
        prediction, answer = prediction[..., valid_cols], answer[..., valid_cols]
        num_valid_qs = valid_cols.long().sum().item()
        idx_order = torch.argsort(prediction, dim=0).to(answer.device).view(-1) * num_valid_qs + torch.arange(num_valid_qs).to(answer.device).repeat(num_items)
        ordered_answer = (answer.view(-1)[idx_order]).view(num_items, num_valid_qs)
        num_pos = torch.cumsum(ordered_answer, dim=0)
        num_neg = torch.arange(num_items).unsqueeze(-1).to(answer.device) - num_pos
        rocauc_scores = 1. - ((num_pos * (ordered_answer == 0)).sum(0) / (num_pos[-1] * num_neg[-1] + 1e-6))
        return rocauc_scores.mean().item()

class HitsEvaluator(BaseEvaluator):
    r"""
        The evaluator for computing Hits@K. This module inputs K, instead of task_ids as the second parameter.
        
        Bases: ``BaseEvaluator``
    """
    def __init__(self, num_tasks, k):
        super().__init__(num_tasks, None)
        self.k = k
    
    def __call__(self, _prediction, _answer, task_ids):
        prediction = _prediction.squeeze().to(_answer.device)
        answer = _answer.squeeze()
        
        neg_samples = prediction[answer == 0]
        if neg_samples.shape[0] < self.k: return torch.ones(self.num_tasks + 1)
        neg_threshold = torch.topk(neg_samples, self.k).values[-1]
        
        num_pos = torch.bincount(task_ids[answer == 1], minlength=self.num_tasks).float()
        num_hits = torch.bincount(task_ids[(answer == 1) & (prediction > neg_threshold)], minlength=self.num_tasks).float()
        
        hits_per_task = torch.zeros(self.num_tasks + 1)
        hits_per_task[:self.num_tasks] = num_hits / torch.clamp(num_pos, min=1.)
        hits_per_task[self.num_tasks] = num_hits.sum() / num_pos.sum()
        return hits_per_task
        
    def simple_eval(self, prediction, answer):
        prediction = _prediction.squeeze().to(_answer.device)
        answer = _answer.squeeze()
        
        neg_samples = prediction[answer == 0]
        if neg_samples.shape[0] < self.k: return torch.ones(self.num_tasks + 1)
        neg_threshold = torch.topk(neg_samples, self.k).values[-1]
        
        num_pos = (answer == 1).float().sum()
        num_hits = ((answer == 1) & (prediction > neg_threshold)).float().sum()
        
        return (num_hits / num_pos).item()