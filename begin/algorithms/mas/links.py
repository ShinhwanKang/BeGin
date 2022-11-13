import sys
import numpy as np
import torch
import copy
import torch.nn.functional as F
from begin.trainers.links import LCTrainer, LPTrainer

class LCTaskILMASTrainer(LCTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.lamb = kwargs['lamb'] if 'lamb' in kwargs else 1.
        
    def prepareLoader(self, _curr_dataset, curr_training_states):
        curr_dataset = copy.deepcopy(_curr_dataset)
        srcs, dsts = curr_dataset.edges()
        labels = curr_dataset.edata.pop('label')
        train_mask = curr_dataset.edata.pop('train_mask')
        val_mask = curr_dataset.edata.pop('val_mask')
        test_mask = curr_dataset.edata.pop('test_mask')
        task_mask = curr_dataset.edata.pop('task_specific_mask')
        curr_dataset = dgl.add_self_loop(curr_dataset)
        return [(curr_dataset, srcs[train_mask], dsts[train_mask], task_mask[train_mask], labels[train_mask])], [(curr_dataset, srcs[val_mask], dsts[val_mask], task_mask[val_mask], labels[val_mask])], [(curr_dataset, srcs[test_mask], dsts[test_mask], task_mask[test_mask], labels[test_mask])]
    
    def inference(self, model, _curr_batch, training_states):
        curr_batch, srcs, dsts, task_masks, labels = _curr_batch
        preds = model(curr_batch.to(self.device), curr_batch.ndata['feat'].to(self.device), srcs, dsts, task_masks=task_masks)
        loss = self.loss_fn(preds, labels.to(self.device))
        return {'preds': preds, 'loss': loss}
    
    def afterInference(self, results, model, optimizer, _curr_batch, training_states):
        loss_reg = 0
        for name, p in model.named_parameters():
            l = self.lamb * training_states['importances'][name]
            l = l * ((p - training_states['params'][name]) ** 2)
            loss_reg = loss_reg + l.sum()
        total_loss = results['loss'] + loss_reg
        total_loss.backward()
        optimizer.step()
        return {'loss': total_loss.item(), 'acc': self.eval_fn(torch.argmax(results['preds'], -1), _curr_batch[-1].to(self.device))}
    
    def initTrainingStates(self, scenario, model, optimizer):
        return {'importances': {name: torch.zeros_like(p) for name, p in model.named_parameters()}, 'params': {name: torch.zeros_like(p) for name, p in model.named_parameters()}}
    
    def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        super().processAfterTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)
        importances = {name: torch.zeros_like(p) for name, p in curr_model.named_parameters()}
        train_loader = self.prepareLoader(curr_dataset, curr_training_states)[0]
        
        total_num_items = 0
        for i, _curr_batch in enumerate(iter(train_loader)):
            curr_model.zero_grad()
            curr_results = self.inference(curr_model, _curr_batch, curr_training_states)
            (torch.linalg.norm(curr_results['preds'], dim=-1) ** 2).mean().backward()
            curr_num_items =_curr_batch[1].shape[0]
            total_num_items += curr_num_items
            for name, p in curr_model.named_parameters():
                curr_training_states['params'][name] = p.data.clone().detach()
                importances[name] += p.grad.data.clone().detach().abs() * curr_num_items
                
        for name, p in curr_model.named_parameters():
            curr_training_states['importances'][name] += (importances[name] / total_num_items)
    
class LCClassILMASTrainer(LCTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.lamb = kwargs['lamb'] if 'lamb' in kwargs else 1.
        
    def inference(self, results, model, optimizer, _curr_batch, training_states):
        loss_reg = 0
        for name, p in model.named_parameters():
            l = self.lamb * training_states['importances'][name]
            l = l * ((p - training_states['params'][name]) ** 2)
            loss_reg = loss_reg + l.sum()
        total_loss = results['loss'] + loss_reg
        total_loss.backward()
        optimizer.step()
        return {'loss': total_loss.item(), 'acc': self.eval_fn(torch.argmax(results['preds'], -1), _curr_batch[-1].to(self.device))}
    
    def initTrainingStates(self, scenario, model, optimizer):
        return {'importances': {name: torch.zeros_like(p) for name, p in model.named_parameters()}, 'params': {name: torch.zeros_like(p) for name, p in model.named_parameters()}}
    
    def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        super().processAfterTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)
        importances = {name: torch.zeros_like(p) for name, p in curr_model.named_parameters()}
        train_loader = self.prepareLoader(curr_dataset, curr_training_states)[0]
        
        total_num_items = 0
        for i, _curr_batch in enumerate(iter(train_loader)):
            curr_model.zero_grad()
            curr_results = self.inference(curr_model, _curr_batch, curr_training_states)
            (torch.linalg.norm(curr_results['preds'], dim=-1) ** 2).mean().backward()
            curr_num_items =_curr_batch[1].shape[0]
            total_num_items += curr_num_items
            for name, p in curr_model.named_parameters():
                curr_training_states['params'][name] = p.data.clone().detach()
                importances[name] += p.grad.data.clone().detach().abs() * curr_num_items
                
        for name, p in curr_model.named_parameters():
            curr_training_states['importances'][name] += (importances[name] / total_num_items)

class LCTimeILMASTrainer(LCTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.lamb = kwargs['lamb'] if 'lamb' in kwargs else 1.
        
    def processBeforeTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        curr_training_states['scheduler'] = self.scheduler_fn(curr_optimizer)
        curr_training_states['best_val_acc'] = -1.
        curr_training_states['best_val_loss'] = 1e10
        curr_model.observe_labels(torch.LongTensor([0]))
        self._reset_optimizer(curr_optimizer)
    
    def processEvalIteration(self, model, _curr_batch):
        results = self._model_inference(model, _curr_batch, None)
        return results['preds'], {'loss': results['loss'].item()}
    
    def afterInference(self, results, model, optimizer, _curr_batch, training_states):
        loss_reg = 0
        for name, p in model.named_parameters():
            l = self.lamb * training_states['importances'][name]
            l = l * ((p - training_states['params'][name]) ** 2)
            loss_reg = loss_reg + l.sum()
        total_loss = results['loss'] + loss_reg
        total_loss.backward()
        optimizer.step()
        return {'loss': total_loss.item(), 'acc': self.eval_fn(results['preds'], _curr_batch[-1].to(self.device))}
    
    def initTrainingStates(self, scenario, model, optimizer):
        return {'importances': {name: torch.zeros_like(p) for name, p in model.named_parameters()}, 'params': {name: torch.zeros_like(p) for name, p in model.named_parameters()}}
    
    def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        super().processAfterTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)
        importances = {name: torch.zeros_like(p) for name, p in curr_model.named_parameters()}
        train_loader = self.prepareLoader(curr_dataset, curr_training_states)[0]
        
        total_num_items = 0
        for i, _curr_batch in enumerate(iter(train_loader)):
            curr_model.zero_grad()
            curr_results = self.inference(curr_model, _curr_batch, curr_training_states)
            (torch.linalg.norm(curr_results['preds'], dim=-1) ** 2).mean().backward()
            curr_num_items =_curr_batch[1].shape[0]
            total_num_items += curr_num_items
            for name, p in curr_model.named_parameters():
                curr_training_states['params'][name] = p.data.clone().detach()
                importances[name] += p.grad.data.clone().detach().abs() * curr_num_items
                
        for name, p in curr_model.named_parameters():
            curr_training_states['importances'][name] += (importances[name] / total_num_items)
        
class LPTimeILMASTrainer(LPTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.lamb = kwargs['lamb'] if 'lamb' in kwargs else 1.
        
    def processTrainIteration(self, model, optimizer, _curr_batch, training_states):
        graph, feats = map(lambda x: x.to(self.device), training_states['graph'])
        edges, labels = map(lambda x: x.to(self.device), _curr_batch)
        optimizer.zero_grad()
        srcs, dsts = edges[:, 0], edges[:, 1]
        neg_dsts = torch.randint(low=0, high=graph.num_nodes(), size=(srcs.shape[0],)).to(self.device)
        preds = model(graph, feats, srcs.repeat(2), torch.cat((edges[:, 1], neg_dsts), dim=0)).squeeze(-1)
        loss = self.loss_fn(preds, torch.cat((labels, torch.zeros_like(labels)), dim=0))
        
        loss_reg = 0
        for name, p in model.named_parameters():
            l = self.lamb * training_states['importances'][name]
            l = l * ((p - training_states['params'][name]) ** 2)
            loss_reg = loss_reg + l.sum()
        
        # print(loss, loss_reg)
        loss = loss + loss_reg
        
        loss.backward()
        optimizer.step()
        return {'_num_items': preds.shape[0], 'loss': loss.item()}
    
    def initTrainingStates(self, scenario, model, optimizer):
        return {'importances': {name: torch.zeros_like(p) for name, p in model.named_parameters()}, 'params': {name: torch.zeros_like(p) for name, p in model.named_parameters()}}
    
    def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        super().processAfterTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)
        params = {}
        importances = {}
        train_loader = self.prepareLoader(curr_dataset, curr_training_states)[0]
        total_num_items = 0
        for i, _curr_batch in enumerate(iter(train_loader)):
            curr_model.zero_grad()
            graph, feats = map(lambda x: x.to(self.device), curr_training_states['graph'])
            edges, labels = map(lambda x: x.to(self.device), _curr_batch)
            srcs, dsts = edges[:, 0], edges[:, 1]
            neg_dsts = torch.randint(low=0, high=graph.num_nodes(), size=(srcs.shape[0],)).to(self.device)
            preds = curr_model(graph, feats, srcs.repeat(2), torch.cat((edges[:, 1], neg_dsts), dim=0)).squeeze(-1)
            (preds ** 2).mean().backward()
            total_num_items += labels.shape[0]
            if i == 0:
                for name, p in curr_model.named_parameters():
                    curr_training_states['params'][name] = p.data.clone().detach()
                    importances[name] = p.grad.data.clone().detach().abs() * labels.shape[0]
            else:
                for name, p in curr_model.named_parameters():
                    importances[name] += p.grad.data.clone().detach().abs() * labels.shape[0]
                    
        for name, p in curr_model.named_parameters():
            curr_training_states['importances'][name] += (importances[name] / total_num_items)

class LPDomainILMASTrainer(LPTimeILMASTrainer):
    pass