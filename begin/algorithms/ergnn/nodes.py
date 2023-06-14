import torch
import dgl
import torch.nn.functional as F
from begin.trainers.nodes import NCTrainer, NCMinibatchTrainer
from .utils import *

class NCTaskILERGNNTrainer(NCTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        """
            `num_experience_nodes` is the hyperparameter for size of the memory.
            `sampler_name` is the hyperparamter for selecting the sampler.
            `distance_threshold` is the hyperparameter about distances of embeddings used in the sampler.
        """
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.num_experience_nodes = kwargs['num_experience_nodes'] if 'num_experience_nodes' in kwargs else 1
        self.sampler_name = kwargs['sampler_name']  if 'sampler_name' in kwargs else 'CM'
        self.distance_threshold = kwargs['distance_threshold'] if 'distance_threshold' in kwargs else 0.5
        
    def initTrainingStates(self, scenario, model, optimizer):
        """
            ERGNN requires node sampler. We use CM sampler as the default sampler.
            
            Args:
                scenario (begin.scenarios.common.BaseScenarioLoader): the given ScenarioLoader to the trainer
                model (torch.nn.Module): the given model to the trainer
                optmizer (torch.optim.Optimizer): the optimizer generated from the given `optimizer_fn` 
                
            Returns:
                Initialized training state (dict).
        """
        sampler_list = {'CM': CM_sampler, 'MF': MF_sampler, 'random': random_sampler}
        _samp = sampler_list[self.sampler_name.split('_')[0]](plus = ('_plus' in self.sampler_name)) 
        return {'sampler': _samp, 'buffered_nodes':[]}
    
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
        return {'preds': preds[mask], 'loss': loss, 'preds_full': preds}
    
    def afterInference(self, results, model, optimizer, _curr_batch, training_states):
        """
            The event function to execute some processes right after the inference step (for training).
            We recommend performing backpropagation in this event function.
            
            ERGNN additionally computes the loss from the buffered nodes and applies it to backpropagation.
            
            Args:
                results (dict): the returned dictionary from the event function `inference`.
                model (torch.nn.Module): the current trained model.
                optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
                
            Returns:
                A dictionary containing the information from the `results`.
        """
        loss = results['loss']
        curr_batch, mask = _curr_batch
        if len(training_states['buffered_nodes'])>0:
            buffer_size = len(training_states['buffered_nodes'])
            beta = buffer_size/(buffer_size+mask.sum())
            buffered_mask = torch.zeros(curr_batch.ndata['feat'].shape[0]).to(self.device)
            buffered_mask[training_states['buffered_nodes']] = 1.
            buffered_mask = buffered_mask.to(torch.bool)
            buffered_loss = self.loss_fn(results['preds_full'][buffered_mask.cpu()].to(self.device), _curr_batch[0].ndata['label'][buffered_mask.cpu()].to(self.device))
            loss = (1-beta) * loss + beta * buffered_loss
        loss.backward()
        optimizer.step()
        return {'loss': loss.item(),
                'acc': self.eval_fn(results['preds'].argmax(-1), curr_batch.ndata['label'][mask].to(self.device))}
    
    def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        """
            The event function to execute some processes after training the current task.
            
            ERGNN selects nodes using the sampler and stores them in the buffer.
                
            Args:
                task_id (int): the index of the current task.
                curr_dataset (object): The dataset for the current task.
                curr_model (torch.nn.Module): the current trained model.
                curr_optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_training_states (dict): the dictionary containing the current training states.
        """
        super().processAfterTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)
        train_nodes = torch.nonzero(curr_dataset.ndata['train_mask']).squeeze().tolist()
        train_classes = curr_dataset.ndata['label'][curr_dataset.ndata['train_mask']].tolist()
        train_ids_per_class = dict()
        for c in set(train_classes):
            train_ids_per_class[c] = list()
        for _idx, c in enumerate(train_classes):
            train_ids_per_class[c].append(train_nodes[_idx])
        ids_per_cls_train = [x for x in train_ids_per_class.values()]
        sampled_ids = curr_training_states['sampler'](ids_per_cls_train, self.num_experience_nodes, curr_dataset.ndata['feat'], curr_model.last_h, self.distance_threshold)
        curr_training_states['buffered_nodes'].extend(sampled_ids)

class NCClassILERGNNTrainer(NCTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        """
            `num_experience_nodes` is the hyperparameter for size of the memory.
            `sampler_name` is the hyperparamter for selecting the sampler.
            `distance_threshold` is the hyperparameter about distances of embeddings used in the sampler.
        """
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.num_experience_nodes = kwargs['num_experience_nodes'] if 'num_experience_nodes' in kwargs else 1
        self.sampler_name = kwargs['sampler_name']  if 'sampler_name' in kwargs else 'CM'
        self.distance_threshold = kwargs['distance_threshold'] if 'distance_threshold' in kwargs else 0.5
        
    def initTrainingStates(self, scenario, model, optimizer):
        """
            ERGNN requires node sampler. We use CM sampler as the default sampler.
            
            Args:
                scenario (begin.scenarios.common.BaseScenarioLoader): the given ScenarioLoader to the trainer
                model (torch.nn.Module): the given model to the trainer
                optmizer (torch.optim.Optimizer): the optimizer generated from the given `optimizer_fn` 
                
            Returns:
                Initialized training state (dict).
        """
        sampler_list = {'CM': CM_sampler, 'MF': MF_sampler, 'random': random_sampler}
        _samp = sampler_list[self.sampler_name.split('_')[0]](plus = ('_plus' in self.sampler_name)) 
        return {'sampler': _samp, 'buffered_nodes':[]}
    
    def inference(self, model, _curr_batch, training_states):
        curr_batch, mask = _curr_batch
        preds = model(curr_batch.to(self.device), curr_batch.ndata['feat'].to(self.device))
        loss = self.loss_fn(preds[mask], curr_batch.ndata['label'][mask].to(self.device))
        return {'preds': preds[mask], 'loss': loss, 'preds_full': preds}
    
    def afterInference(self, results, model, optimizer, _curr_batch, training_states):
        """
            The event function to execute some processes right after the inference step (for training).
            We recommend performing backpropagation in this event function.
            
            ERGNN additionally computes the loss from the buffered nodes and applies it to backpropagation.
            
            Args:
                results (dict): the returned dictionary from the event function `inference`.
                model (torch.nn.Module): the current trained model.
                optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
                
            Returns:
                A dictionary containing the information from the `results`.
        """
        loss = results['loss']
        curr_batch, mask = _curr_batch
        if len(training_states['buffered_nodes'])>0:
            buffer_size = len(training_states['buffered_nodes'])
            beta = buffer_size/(buffer_size+mask.sum())
            buffered_mask = torch.zeros(curr_batch.ndata['feat'].shape[0]).to(self.device)
            buffered_mask[training_states['buffered_nodes']] = 1.
            buffered_mask = buffered_mask.to(torch.bool)
            buffered_loss = self.loss_fn(results['preds_full'][buffered_mask.cpu()].to(self.device), _curr_batch[0].ndata['label'][buffered_mask.cpu()].to(self.device))
            loss = (1-beta) * loss + beta * buffered_loss
        loss.backward()
        optimizer.step()
        return {'loss': loss.item(),
                'acc': self.eval_fn(results['preds'].argmax(-1), curr_batch.ndata['label'][mask].to(self.device))}
    
    def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        """
            The event function to execute some processes after training the current task.
            
            ERGNN selects nodes using the sampler and stores them in the buffer.
                
            Args:
                task_id (int): the index of the current task.
                curr_dataset (object): The dataset for the current task.
                curr_model (torch.nn.Module): the current trained model.
                curr_optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_training_states (dict): the dictionary containing the current training states.
        """
        super().processAfterTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)
        train_nodes = torch.nonzero(curr_dataset.ndata['train_mask']).squeeze().tolist()
        train_classes = curr_dataset.ndata['label'][curr_dataset.ndata['train_mask']].tolist()
        train_ids_per_class = dict()
        for c in set(train_classes):
            train_ids_per_class[c] = list()
        for _idx, c in enumerate(train_classes):
            train_ids_per_class[c].append(train_nodes[_idx])
        ids_per_cls_train = [x for x in train_ids_per_class.values()]
        sampled_ids = curr_training_states['sampler'](ids_per_cls_train, self.num_experience_nodes, curr_dataset.ndata['feat'], curr_model.last_h, self.distance_threshold)
        curr_training_states['buffered_nodes'].extend(sampled_ids)

class NCClassILERGNNMinibatchTrainer(NCMinibatchTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        """
            `num_experience_nodes` is the hyperparameter for size of the memory.
            `sampler_name` is the hyperparamter for selecting the sampler.
            `distance_threshold` is the hyperparameter about distances of embeddings used in the sampler.
        """
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.num_experience_nodes = kwargs['num_experience_nodes'] if 'num_experience_nodes' in kwargs else 1
        self.sampler_name = kwargs['sampler_name']  if 'sampler_name' in kwargs else 'CM'
        self.distance_threshold = kwargs['distance_threshold'] if 'distance_threshold' in kwargs else 0.5
        
    def initTrainingStates(self, scenario, model, optimizer):
        """
            ERGNN requires node sampler. We use CM sampler as the default sampler.
            
            Args:
                scenario (begin.scenarios.common.BaseScenarioLoader): the given ScenarioLoader to the trainer
                model (torch.nn.Module): the given model to the trainer
                optmizer (torch.optim.Optimizer): the optimizer generated from the given `optimizer_fn` 
                
            Returns:
                Initialized training state (dict).
        """
        sampler_list = {'CM': CM_sampler, 'MF': MF_sampler, 'random': random_sampler}
        _samp = sampler_list[self.sampler_name.split('_')[0]](plus = ('_plus' in self.sampler_name)) 
        return {'sampler': _samp, 'buffered_nodes':[]}
    
    def inference(self, model, _curr_batch, training_states):
        input_nodes, output_nodes, blocks = _curr_batch
        blocks = [b.to(self.device) for b in blocks]
        labels = blocks[-1].dstdata['label']
        preds = model.bforward(blocks, blocks[0].srcdata['feat'])
        if training_states is not None: training_states['saved_feats'][output_nodes] = model.last_h.detach().cpu()
        loss = self.loss_fn(preds, labels)
        return {'preds': preds, 'loss': loss}
    
    def processBeforeTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        super().processBeforeTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)
        curr_training_states['saved_feats'] = torch.zeros(curr_dataset.num_nodes(), curr_model.n_hidden)
        
    def afterInference(self, results, model, optimizer, _curr_batch, training_states):
        """
            The event function to execute some processes right after the inference step (for training).
            We recommend performing backpropagation in this event function.
            
            ERGNN additionally computes the loss from the buffered nodes and applies it to backpropagation.
            
            Args:
                results (dict): the returned dictionary from the event function `inference`.
                model (torch.nn.Module): the current trained model.
                optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
                
            Returns:
                A dictionary containing the information from the `results`.
        """
        loss = results['loss']
        if len(training_states['buffered_nodes'])>0:
            buffer_size = len(training_states['buffered_nodes'])
            beta = buffer_size/(buffer_size+results['preds'].shape[0])
            for _buf_batch in training_states['buffered_loader']:
                buf_results = self.inference(model, _buf_batch, training_states)
            buffered_loss = buf_results['loss']
            loss = (1-beta) * loss + beta * buffered_loss
        loss.backward()
        optimizer.step()
        return {'loss': loss.item(),
                'acc': self.eval_fn(results['preds'].argmax(-1), _curr_batch[-1][-1].dstdata['label'].to(self.device))}
    
    def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        """
            The event function to execute some processes after training the current task.
            
            ERGNN selects nodes using the sampler and stores them in the buffer.
                
            Args:
                task_id (int): the index of the current task.
                curr_dataset (object): The dataset for the current task.
                curr_model (torch.nn.Module): the current trained model.
                curr_optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_training_states (dict): the dictionary containing the current training states.
        """
        super().processAfterTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)
        train_nodes = torch.nonzero(curr_dataset.ndata['train_mask']).squeeze().tolist()
        train_classes = curr_dataset.ndata['label'][curr_dataset.ndata['train_mask']].tolist()
        train_ids_per_class = dict()
        for c in set(train_classes):
            train_ids_per_class[c] = list()
        for _idx, c in enumerate(train_classes):
            train_ids_per_class[c].append(train_nodes[_idx])
        ids_per_cls_train = [x for x in train_ids_per_class.values()]
        sampled_ids = curr_training_states['sampler'](ids_per_cls_train, self.num_experience_nodes, curr_dataset.ndata['feat'], curr_training_states['saved_feats'], self.distance_threshold)
        curr_training_states['buffered_nodes'].extend(sampled_ids)
        
        g_buf = torch.Generator()
        g_buf.manual_seed(0)
        buf_sampler = dgl.dataloading.MultiLayerNeighborSampler([5, 10, 10])
        buf_loader = dgl.dataloading.NodeDataLoader(
            curr_dataset, curr_training_states['buffered_nodes'], buf_sampler,
            batch_size=131072,
            shuffle=True,
            drop_last=False,
            num_workers=1, worker_init_fn=self._dataloader_seed_worker, generator=g_buf)
        curr_training_states['buffered_loader'] = buf_loader
        
class NCDomainILERGNNTrainer(NCTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        """
            `num_experience_nodes` is the hyperparameter for size of the memory.
            `sampler_name` is the hyperparamter for selecting the sampler.
            `distance_threshold` is the hyperparameter about distances of embeddings used in the sampler.
        """
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.num_experience_nodes = kwargs['num_experience_nodes'] if 'num_experience_nodes' in kwargs else 1
        self.sampler_name = kwargs['sampler_name']  if 'sampler_name' in kwargs else 'CM'
        self.distance_threshold = kwargs['distance_threshold'] if 'distance_threshold' in kwargs else 0.5
        raise NotImplementedError
        
class NCTimeILERGNNTrainer(NCClassILERGNNTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        """
            `num_experience_nodes` is the hyperparameter for size of the memory.
            `sampler_name` is the hyperparamter for selecting the sampler.
            `distance_threshold` is the hyperparameter about distances of embeddings used in the sampler.
        """
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.num_experience_nodes = kwargs['num_experience_nodes'] if 'num_experience_nodes' in kwargs else 1
        self.sampler_name = kwargs['sampler_name']  if 'sampler_name' in kwargs else 'CM'
        self.distance_threshold = kwargs['distance_threshold'] if 'distance_threshold' in kwargs else 0.5
        
    def processTrainIteration(self, model, optimizer, _curr_batch, training_states):
        """
            The event function to handle every training iteration.
            
            ERGNN additionally computes the loss from the buffered nodes and applies it to backpropagation.
            
            Args:
                model (torch.nn.Module): the current trained model.
                optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
                
            Returns:
                A dictionary containing the outcomes (stats) during the training iteration.
        """
        model.train()
        model.zero_grad()
        curr_batch, mask = _curr_batch
        preds = model(curr_batch.to(self.device), curr_batch.ndata['feat'].to(self.device))
        loss = self.loss_fn(preds[mask].to(self.device), curr_batch.ndata['label'][mask].to(self.device))
        
        if len(training_states['buffered_nodes'])>0:
            buffer_size = len(training_states['buffered_nodes'])
            beta = buffer_size/(buffer_size+mask.sum())
            buffered_mask = torch.zeros(curr_batch.ndata['feat'].shape[0]).to(self.device)
            buffered_mask[training_states['buffered_nodes']] = 1.
            buffered_mask = buffered_mask.to(torch.bool)
            buffered_loss = self.loss_fn(preds[buffered_mask.cpu()].to(self.device), curr_batch.ndata['label'][buffered_mask.cpu()].to(self.device))
            loss = (1-beta) * loss + beta * buffered_loss
            
        loss.backward()
        optimizer.step()        
        return {'loss': loss.item(), 'acc': self.eval_fn(preds[mask].argmax(-1), curr_batch.ndata['label'][mask].to(self.device))}
    
    def initTrainingStates(self, scenario, model, optimizer):
        """
            ERGNN requires node sampler. We use CM sampler as the default sampler.
            
            Args:
                scenario (begin.scenarios.common.BaseScenarioLoader): the given ScenarioLoader to the trainer
                model (torch.nn.Module): the given model to the trainer
                optmizer (torch.optim.Optimizer): the optimizer generated from the given `optimizer_fn` 
                
            Returns:
                Initialized training state (dict).
        """
        _samp = None
        if self.sampler_name == 'CM':
            _samp = CM_sampler(plus=False)
        elif self.sampler_name == 'CM_plus':
            _samp = CM_sampler(plus=True)
        elif self.sampler_name == 'MF':
            _samp = MF_sampler(plus=False)
        elif self.sampler_name == 'MF_plus':
            _samp = MF_sampler(plus=True)
        elif self.sampler_name == 'random':
            _samp = random_sampler(plus=False)
        return {'sampler': _samp, 'buffered_nodes':[]}
    
    def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        """
            The event function to execute some processes after training the current task.
            
            ERGNN selects nodes using the sampler and stores them in the buffer.
                
            Args:
                task_id (int): the index of the current task.
                curr_dataset (object): The dataset for the current task.
                curr_model (torch.nn.Module): the current trained model.
                curr_optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_training_states (dict): the dictionary containing the current training states.
        """
        super().processAfterTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)
        train_nodes = torch.nonzero(curr_dataset.ndata['train_mask']).squeeze().tolist()
        train_classes = curr_dataset.ndata['label'][curr_dataset.ndata['train_mask']].tolist()
        train_ids_per_class = dict()
        for c in set(train_classes):
            train_ids_per_class[c] = list()
        for _idx, c in enumerate(train_classes):
            train_ids_per_class[c].append(train_nodes[_idx])
        ids_per_cls_train = [x for x in train_ids_per_class.values()]
        sampled_ids = curr_training_states['sampler'](ids_per_cls_train, self.num_experience_nodes, curr_dataset.ndata['feat'], curr_model.last_h, self.distance_threshold, incr_type='time')
        curr_training_states['buffered_nodes'].extend(sampled_ids)
