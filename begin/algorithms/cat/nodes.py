import sys
import torch
import dgl
from begin.trainers.nodes import NCTrainer, NCMinibatchTrainer
import copy
import torch.nn.functional as F
from torch import nn

class NCTaskILCaTTrainer(NCTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.num_memories = kwargs['num_memories'] if 'num_memories' in kwargs else 100
        self.num_memories = (self.num_memories // self.num_tasks)
        
    def processBeforeTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        super().processBeforeTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)
        init_model = copy.deepcopy(curr_model)
        self._reset_model(init_model)
        new_condensed_y = []
        
        gt_x = curr_dataset.ndata['feat'][curr_dataset.ndata['train_mask']]
        gt_y = curr_dataset.ndata['label'][curr_dataset.ndata['train_mask']].squeeze()
        gt_mask = curr_dataset.ndata['task_specific_mask'][curr_dataset.ndata['train_mask']]
        num_target_nodes = gt_y.shape[0]
        candidates = []
        gt_bincount = torch.bincount(gt_y).tolist()
        
        for class_id, cnt in enumerate(gt_bincount):
            if cnt > 0:
                new_condensed_y = new_condensed_y + [class_id for _ in range(max(1, (cnt * self.num_memories) // num_target_nodes))]
                if cnt * self.num_memories >= num_target_nodes:
                    candidates.append(( ( (cnt * self.num_memories) % num_target_nodes ) / num_target_nodes , class_id))
        candidates = sorted(candidates, reverse=True)
        if len(new_condensed_y) < self.num_memories:
            # keep balance as possible
            new_condensed_y += [k for v, k in candidates[:self.num_memories - len(new_condensed_y)]]
        
        new_condensed_y = torch.LongTensor(new_condensed_y)
        new_condensed_x = torch.zeros(len(new_condensed_y), gt_x.shape[-1])
        new_condensed_mask = torch.zeros(len(new_condensed_y), curr_dataset.ndata['task_specific_mask'].shape[-1], dtype=torch.bool)
        # initialize condensed feature
        condensed_bincount = torch.bincount(new_condensed_y).tolist()
        for class_id, cnt in enumerate(condensed_bincount):
            if cnt > 0:
                perm = torch.randperm(gt_bincount[class_id])
                new_condensed_x[new_condensed_y == class_id] = (gt_x[gt_y == class_id][perm])[:cnt]
                new_condensed_mask[new_condensed_y == class_id] = (gt_mask[gt_y == class_id][perm])[:cnt]
                
        new_condensed_x = nn.Parameter(new_condensed_x)
        
        pre_optimizer = self.optimizer_fn([new_condensed_x])
        # pre_scheduler = self.scheduler_fn(pre_optimizer)
        trainloader, _1, _2 = self.prepareLoader(curr_dataset, curr_training_states)
        
        best_val_loss = 1e10
        for epoch_cnt in range(self.max_num_epochs):
            init_model.reset_parameters()
            for _curr_batch in trainloader:
                pre_optimizer.zero_grad()
                curr_batch, mask = _curr_batch
                real_output = F.normalize(init_model.forward_without_classifier(curr_batch.to(self.device), curr_batch.ndata['feat'].to(self.device))[mask], dim=-1)
                fake_output = F.normalize(init_model.forward_without_classifier(None, new_condensed_x.to(self.device)), dim=-1)

                _loss = 0.
                for class_id, cnt in enumerate(gt_bincount):
                    if cnt > 0:
                        _loss = _loss + (cnt / num_target_nodes) * ((real_output[gt_y == class_id].mean(0) - fake_output[new_condensed_y == class_id].mean(0)) ** 2).sum()
                        
                # _loss = ((real_output.mean(0) - fake_output.mean(0)) ** 2).sum(-1).mean(0)
                _loss.backward()
                pre_optimizer.step()
            """
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                pre_checkpoint = copy.deepcopy(new_condensed_x.data.detach())
            pre_scheduler.step(val_loss)
            if -1e-9 < (pre_optimizer.param_groups[0]['lr'] - pre_scheduler.min_lrs[0]) < 1e-9:
                break
            """
        # with torch.no_grad():
        #     new_condensed_x.data.copy_(pre_checkpoint)
        curr_training_states['memory_x'].append(new_condensed_x)
        curr_training_states['memory_y'].append(new_condensed_y)
        curr_training_states['memory_mask'].append(new_condensed_mask)
        curr_training_states['memory_weight'].append(torch.ones(len(new_condensed_y)) * num_target_nodes / len(new_condensed_y))
        
    def initTrainingStates(self, scenario, model, optimizer):
        return {'memory_x': [], 'memory_y': [], 'memory_mask': [], 'memory_weight': []}

    def processTrainIteration(self, model, optimizer, _curr_batch, training_states):
        optimizer.zero_grad()
        preds = model(None, torch.cat(training_states['memory_x'], dim=0).to(self.device), task_masks=torch.cat(training_states['memory_mask'], dim=0).to(self.device))
        loss = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none')(preds, torch.cat(training_states['memory_y'], dim=0).to(self.device))
        loss = (torch.cat(training_states['memory_weight'], dim=0).to(self.device) * loss).sum()
        loss.backward()
        optimizer.step()
        return {'loss': loss.item(), 'acc': self.eval_fn(self.predictionFormat({'preds': preds}), torch.cat(training_states['memory_y'], dim=0).to(self.device))}
        
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
        return {'preds': preds[mask], 'loss': loss}

    def afterInference(self, results, model, optimizer, _curr_batch, training_states):
        results['loss'].backward()
        optimizer.step()
        return {'loss': results['loss'].item(), 'acc': self.eval_fn(self.predictionFormat(results), _curr_batch[0].ndata['label'][_curr_batch[1]].to(self.device))}

class NCClassILCaTTrainer(NCTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.num_memories = kwargs['num_memories'] if 'num_memories' in kwargs else 100
        self.num_memories = (self.num_memories // self.num_tasks)
        
    def processBeforeTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        super().processBeforeTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)
        init_model = copy.deepcopy(curr_model)
        self._reset_model(init_model)
        new_condensed_y = []
        
        gt_x = curr_dataset.ndata['feat'][curr_dataset.ndata['train_mask']]
        gt_y = curr_dataset.ndata['label'][curr_dataset.ndata['train_mask']].squeeze()
        # gt_mask = curr_dataset.ndata['task_specific_mask'][curr_dataset.ndata['train_mask']]
        num_target_nodes = gt_y.shape[0]
        candidates = []
        gt_bincount = torch.bincount(gt_y).tolist()
        
        for class_id, cnt in enumerate(gt_bincount):
            if cnt > 0:
                new_condensed_y = new_condensed_y + [class_id for _ in range(max(1, (cnt * self.num_memories) // num_target_nodes))]
                if cnt * self.num_memories >= num_target_nodes:
                    candidates.append(( ( (cnt * self.num_memories) % num_target_nodes ) / num_target_nodes , class_id))
        candidates = sorted(candidates, reverse=True)
        if len(new_condensed_y) < self.num_memories:
            # keep balance as possible
            new_condensed_y += [k for v, k in candidates[:self.num_memories - len(new_condensed_y)]]
        
        new_condensed_y = torch.LongTensor(new_condensed_y)
        new_condensed_x = torch.zeros(len(new_condensed_y), gt_x.shape[-1])
        # new_condensed_mask = torch.zeros(len(new_condensed_y), curr_dataset.ndata['task_specific_mask'].shape[-1], dtype=torch.bool)
        print(new_condensed_y)
        # initialize condensed feature
        condensed_bincount = torch.bincount(new_condensed_y).tolist()
        for class_id, cnt in enumerate(condensed_bincount):
            if cnt > 0:
                perm = torch.randperm(gt_bincount[class_id])
                new_condensed_x[new_condensed_y == class_id] = (gt_x[gt_y == class_id][perm])[:cnt]
                # new_condensed_mask[new_condensed_y == class_id] = (gt_mask[gt_y == class_id][perm])[:cnt]
                
        new_condensed_x = nn.Parameter(new_condensed_x)
        
        pre_optimizer = self.optimizer_fn([new_condensed_x])
        # pre_scheduler = self.scheduler_fn(pre_optimizer)
        trainloader, _1, _2 = self.prepareLoader(curr_dataset, curr_training_states)
        
        best_val_loss = 1e10
        for epoch_cnt in range(self.max_num_epochs):
            init_model.reset_parameters()
            for _curr_batch in trainloader:
                pre_optimizer.zero_grad()
                curr_batch, mask = _curr_batch
                real_output = F.normalize(init_model.forward_without_classifier(curr_batch.to(self.device), curr_batch.ndata['feat'].to(self.device))[mask], dim=-1)
                fake_output = F.normalize(init_model.forward_without_classifier(None, new_condensed_x.to(self.device)), dim=-1)

                _loss = 0.
                for class_id, cnt in enumerate(gt_bincount):
                    if cnt > 0:
                        _loss = _loss + (cnt / num_target_nodes) * ((real_output[gt_y == class_id].mean(0) - fake_output[new_condensed_y == class_id].mean(0)) ** 2).sum()
                        
                # _loss = ((real_output.mean(0) - fake_output.mean(0)) ** 2).sum(-1).mean(0)
                _loss.backward()
                pre_optimizer.step()
            """
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                pre_checkpoint = copy.deepcopy(new_condensed_x.data.detach())
            pre_scheduler.step(val_loss)
            if -1e-9 < (pre_optimizer.param_groups[0]['lr'] - pre_scheduler.min_lrs[0]) < 1e-9:
                break
            """
        # with torch.no_grad():
        #     new_condensed_x.data.copy_(pre_checkpoint)
        curr_training_states['memory_x'].append(new_condensed_x)
        curr_training_states['memory_y'].append(new_condensed_y)
        curr_training_states['memory_weight'].append(torch.ones(len(new_condensed_y)) * num_target_nodes / len(new_condensed_y))
        
    def initTrainingStates(self, scenario, model, optimizer):
        return {'memory_x': [], 'memory_y': [], 'memory_weight': []}

    def processTrainIteration(self, model, optimizer, _curr_batch, training_states):
        optimizer.zero_grad()
        preds = model(None, torch.cat(training_states['memory_x'], dim=0).to(self.device))
        loss = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none')(preds, torch.cat(training_states['memory_y'], dim=0).to(self.device))
        weight = torch.cat(training_states['memory_weight'], dim=0).to(self.device)
        weight = weight / weight.sum()
        loss = (weight * loss).sum()
        loss.backward()
        optimizer.step()
        return {'loss': loss.item(), 'acc': self.eval_fn(self.predictionFormat({'preds': preds}), torch.cat(training_states['memory_y'], dim=0).to(self.device))}
        
class NCDomainILCaTTrainer(NCClassILCaTTrainer):
    """
        This trainer has the same behavior as `NCTrainer`.
    """
    pass
        
class NCTimeILCaTTrainer(NCClassILCaTTrainer):
    """
        This trainer has the same behavior as `NCTrainer`.
    """
    pass