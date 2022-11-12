import sys
import numpy as np
import torch
import copy
import torch.nn.functional as F
from begin.trainers.links import LCTrainer, LPTrainer
from .utils import project2cone2

class LCTaskILGEMTrainer(LCTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.lamb = kwargs['lamb'] if 'lamb' in kwargs else .5
        self.num_memories = kwargs['num_memories'] if 'num_memories' in kwargs else 100
        self.num_memories = (self.num_memories // self.num_tasks)
        
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
    
    def initTrainingStates(self, scenario, model, optimizer):
        return {'memories': []}
    
    def beforeInference(self, model, optimizer, _curr_batch, training_states):
        if len(training_states['memories']) > 0:
            all_grads = []
            for mem in training_states['memories']:
                model.zero_grad()
                mem_batch = (copy.deepcopy(_curr_batch[0]), *mem)
                pre_results = self.inference(model, mem_batch, training_states)
                pre_results['loss'].backward()
                all_grads.append(torch.cat([p.grad.data.clone().view(-1) for p in model.parameters()]))
            training_states['all_grads'] = torch.stack(all_grads, dim=0)
        model.zero_grad()
        
    def afterInference(self, results, model, optimizer, _curr_batch, training_states):
        results['loss'].backward()
        if len(training_states['memories']) > 0:
            curr_grad = torch.cat([p.grad.data.view(-1) for p in model.parameters()])
            if ((training_states['all_grads'] * curr_grad).sum(-1) < 0).any():
                new_gradient = project2cone2(curr_grad, training_states['all_grads'], self.lamb)
                curr_idx = 0
                for p in model.parameters():
                    p_size = p.data.numel()
                    p.grad.copy_(new_gradient[curr_idx:(curr_idx + p_size)].view_as(p.data))
                    curr_idx += p_size
        optimizer.step()
        return {'loss': results['loss'].item(), 'acc': self.eval_fn(torch.argmax(results['preds'], -1), _curr_batch[-1].to(self.device))}
    
    def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        super()._processAfterTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)
        train_loader = self.prepareLoader(curr_dataset, curr_training_states)[0]
        chosen_edges = []
        for i, _curr_batch in enumerate(iter(train_loader)):
            curr_batch, srcs, dsts, tmasks, labels = _curr_batch
            perm = torch.randperm(srcs.shape[0])
            chosen_edges = (srcs[perm[:self.num_memories]], dsts[perm[:self.num_memories]], tmasks[perm[:self.num_memories]], labels[perm[:self.num_memories]])
        curr_training_states['memories'].append(chosen_edges)
    
class LCClassILGEMTrainer(LCTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.lamb = kwargs['lamb'] if 'lamb' in kwargs else .5
        self.num_memories = kwargs['num_memories'] if 'num_memories' in kwargs else 100
        self.num_memories = (self.num_memories // self.num_tasks)
        
    def initTrainingStates(self, scenario, model, optimizer):
        return {'memories': []}
    
    def beforeInference(self, model, optimizer, _curr_batch, training_states):
        if len(training_states['memories']) > 0:
            all_grads = []
            for mem in training_states['memories']:
                model.zero_grad()
                mem_batch = (copy.deepcopy(_curr_batch[0]), *mem)
                pre_results = self.inference(model, mem_batch, training_states)
                pre_results['loss'].backward()
                all_grads.append(torch.cat([p.grad.data.clone().view(-1) for p in model.parameters()]))
            training_states['all_grads'] = torch.stack(all_grads, dim=0)
        model.zero_grad()
        
    def afterInference(self, results, model, optimizer, _curr_batch, training_states):
        results['loss'].backward()
        if len(training_states['memories']) > 0:
            curr_grad = torch.cat([p.grad.data.view(-1) for p in model.parameters()])
            if ((training_states['all_grads'] * curr_grad).sum(-1) < 0).any():
                new_gradient = project2cone2(curr_grad, training_states['all_grads'], self.lamb)
                curr_idx = 0
                for p in model.parameters():
                    p_size = p.data.numel()
                    p.grad.copy_(new_gradient[curr_idx:(curr_idx + p_size)].view_as(p.data))
                    curr_idx += p_size
        optimizer.step()
        return {'loss': results['loss'].item(), 'acc': self.eval_fn(torch.argmax(results['preds'], -1), _curr_batch[-1].to(self.device))}
    
    def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        super().processAfterTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)
        train_loader = self.prepareLoader(curr_dataset, curr_training_states)[0]
        chosen_edges = []
        for i, _curr_batch in enumerate(iter(train_loader)):
            curr_batch, srcs, dsts, labels = _curr_batch
            perm = torch.randperm(srcs.shape[0])
            chosen_edges = (srcs[perm[:self.num_memories]], dsts[perm[:self.num_memories]], labels[perm[:self.num_memories]])
        curr_training_states['memories'].append(chosen_edges)

class LCTimeILGEMTrainer(LCTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.lamb = kwargs['lamb'] if 'lamb' in kwargs else .5
        self.num_memories = kwargs['num_memories'] if 'num_memories' in kwargs else 100
        self.num_memories = (self.num_memories // self.num_tasks)
        
    def processBeforeTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        curr_training_states['scheduler'] = self.scheduler_fn(curr_optimizer)
        curr_training_states['best_val_acc'] = -1.
        curr_training_states['best_val_loss'] = 1e10
        curr_model.observe_labels(torch.LongTensor([0]))
        self._reset_optimizer(curr_optimizer)
    
    def processEvalIteration(self, model, _curr_batch):
        results = self.inference(model, _curr_batch, None)
        return results['preds'], {'loss': results['loss'].item()}
    
    def initTrainingStates(self, scenario, model, optimizer):
        return {'memories': []}
    
    def beforeInference(self, model, optimizer, _curr_batch, training_states):
        if len(training_states['memories']) > 0:
            all_grads = []
            for mem in training_states['memories']:
                model.zero_grad()
                mem_batch = (copy.deepcopy(_curr_batch[0]), *mem)
                pre_results = self.inference(model, mem_batch, training_states)
                pre_results['loss'].backward()
                all_grads.append(torch.cat([p.grad.data.clone().view(-1) for p in model.parameters()]))
            training_states['all_grads'] = torch.stack(all_grads, dim=0)
        model.zero_grad()
        
    def afterInference(self, results, model, optimizer, _curr_batch, training_states):
        results['loss'].backward()
        if len(training_states['memories']) > 0:
            curr_grad = torch.cat([p.grad.data.view(-1) for p in model.parameters()])
            if ((training_states['all_grads'] * curr_grad).sum(-1) < 0).any():
                new_gradient = project2cone2(curr_grad, training_states['all_grads'], self.lamb)
                curr_idx = 0
                for p in model.parameters():
                    p_size = p.data.numel()
                    p.grad.copy_(new_gradient[curr_idx:(curr_idx + p_size)].view_as(p.data))
                    curr_idx += p_size
        optimizer.step()
        return {'loss': results['loss'].item(), 'acc': self.eval_fn(results['preds'], _curr_batch[-1].to(self.device))}
    
    def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        super().processAfterTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)
        train_loader = self.prepareLoader(curr_dataset, curr_training_states)[0]
        chosen_edges = []
        for i, _curr_batch in enumerate(iter(train_loader)):
            curr_batch, srcs, dsts, labels = _curr_batch
            perm = torch.randperm(srcs.shape[0])
            chosen_edges = (srcs[perm[:self.num_memories]], dsts[perm[:self.num_memories]], labels[perm[:self.num_memories]])
        curr_training_states['memories'].append(chosen_edges)
        
class LPDomainILGEMTrainer(LPTimeILGEMTrainer):
    pass

class LPTimeILGEMTrainer(LPTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.lamb = kwargs['lamb'] if 'lamb' in kwargs else .5
        self.num_memories = kwargs['num_memories'] if 'num_memories' in kwargs else 100
        self.num_memories = (self.num_memories // self.num_tasks)
        
    def processTrainIteration(self, model, optimizer, _curr_batch, training_states):
        if len(training_states['memories']) > 0:
            all_grads = []
            for saved_batch in training_states['memories']:
                (_saved_graph, _saved_feats), _saved_edges, _saved_labels = saved_batch
                _saved_graph, _saved_feats, _saved_edges, _saved_labels = map(lambda x: x.to(self.device), (_saved_graph, _saved_feats, _saved_edges, _saved_labels))
                model.zero_grad()
                _srcs, _dsts = _saved_edges[:, 0], _saved_edges[:, 1]
                _neg_dsts = torch.randint(low=0, high=_saved_graph.num_nodes(), size=(_srcs.shape[0],)).to(self.device)
                _preds = model(_saved_graph, _saved_feats, _srcs.repeat(2), torch.cat((_saved_edges[:, 1], _neg_dsts), dim=0)).squeeze(-1)
                loss = self.loss_fn(_preds, torch.cat((_saved_labels, torch.zeros_like(_saved_labels)), dim=0))
                loss.backward()
                all_grads.append(torch.cat([p.grad.data.clone().view(-1) for p in model.parameters()]))
            all_grads = torch.stack(all_grads, dim=0)
            
        graph, feats = map(lambda x: x.to(self.device), training_states['graph'])
        edges, labels = map(lambda x: x.to(self.device), _curr_batch)
        optimizer.zero_grad()
        model.zero_grad()
        srcs, dsts = edges[:, 0], edges[:, 1]
        neg_dsts = torch.randint(low=0, high=graph.num_nodes(), size=(srcs.shape[0],)).to(self.device)
        preds = model(graph, feats, srcs.repeat(2), torch.cat((edges[:, 1], neg_dsts), dim=0)).squeeze(-1)
        loss = self.loss_fn(preds, torch.cat((labels, torch.zeros_like(labels)), dim=0))
        loss.backward()
        
        if len(training_states['memories']) > 0:
            curr_grad = torch.cat([p.grad.data.view(-1) for p in model.parameters()])
            if ((all_grads * curr_grad).sum(-1) < 0).any():
                new_gradient = project2cone2(curr_grad, all_grads, self.lamb)
                curr_idx = 0
                for p in model.parameters():
                    p_size = p.data.numel()
                    p.grad.copy_(new_gradient[curr_idx:(curr_idx + p_size)].view_as(p.data))
                    curr_idx += p_size            
        optimizer.step()
        
        training_states['curr_memory']['edges'].append(edges)
        training_states['curr_memory']['labels'].append(labels)
        training_states['curr_memory']['ranges'].append((0, edges.shape[0]))
        training_states['curr_memory']['size'] += edges.shape[0]
        while training_states['curr_memory']['size'] > self.num_memories:
            _from, _to = training_states['curr_memory']['ranges'][0]
            _diff = training_states['curr_memory']['size'] - self.num_memories
            if _diff >= (_to - _from):
                training_states['curr_memory']['edges'] = training_states['curr_memory']['edges'][1:] 
                training_states['curr_memory']['labels'] = training_states['curr_memory']['labels'][1:]
                training_states['curr_memory']['ranges'] = training_states['curr_memory']['ranges'][1:]
                training_states['curr_memory']['size'] -= (_to - _from)
            else:
                training_states['curr_memory']['ranges'][0] = (_from + _diff, _to)
                training_states['curr_memory']['size'] -= _diff
                break
                
        return {'_num_items': preds.shape[0], 'loss': loss.item()}
    
    def initTrainingStates(self, scenario, model, optimizer):
        return {'memories': []}
    
    def processBeforeTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        super().processBeforeTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)
        curr_training_states['curr_memory'] = {'edges': [], 'labels': [], 'ranges': [], 'size': 0}
        
    def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        super().processAfterTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)
        mem = curr_training_states['curr_memory']
        chosen_edges = torch.cat([edges[_from:_to] for edges, (_from, _to) in zip(mem['edges'], mem['ranges'])], dim=0)
        chosen_labels = torch.cat([labels[_from:_to] for labels, (_from, _to) in zip(mem['labels'], mem['ranges'])], dim=0)
        curr_training_states['memories'].append((copy.deepcopy(curr_training_states['graph']), chosen_edges, chosen_labels))