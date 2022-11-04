from .common import BaseIncrementalBenchmark
import torch

class BaseGraphIncrementalBenchmark(BaseIncrementalBenchmark):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.scheduler_fn = kwargs['scheduler_fn']
        
        # For reproducibility
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        self._dataloader_seed_worker = seed_worker

        
    def prepareLoader(self, curr_dataset, curr_training_states):
        g_train = torch.Generator()
        g_train.manual_seed(0)
        train_loader = dgl.dataloading.GraphDataLoader(curr_dataset['train'], batch_size=128, shuffle=True, drop_last=False, num_workers=4, worker_init_fn=self._dataloader_seed_worker, generator=g_train)
        
        g_val = torch.Generator()
        g_val.manual_seed(0)
        val_loader = dgl.dataloading.GraphDataLoader(curr_dataset['val'], batch_size=128, shuffle=False, drop_last=False, num_workers=4, worker_init_fn=self._dataloader_seed_worker, generator=g_val)
        
        g_test = torch.Generator()
        g_test.manual_seed(0)
        test_loader = dgl.dataloading.GraphDataLoader(curr_dataset['test'], batch_size=128, shuffle=False, drop_last=False, num_workers=4, worker_init_fn=self._dataloader_seed_worker, generator=g_test)
        return train_loader, val_loader, test_loader
    
    def processTrainIteration(self, model, optimizer, _curr_batch, training_states):
        graphs, labels = _curr_batch
        optimizer.zero_grad()
        preds = model(graphs.to(self.device),
                      graphs.ndata['feat'].to(self.device) if 'feat' in graphs.ndata else None,
                      edge_attr = graphs.edata['feat'].to(self.device) if 'feat' in graphs.edata else None,
                      edge_weight = graphs.edata['weight'].to(self.device) if 'weight' in graphs.edata else None)
        loss = self.loss_fn(preds, labels.to(self.device))
        loss.backward()
        optimizer.step()
        return {'_num_items': preds.shape[0], 'loss': loss.item(), 'acc': self.eval_fn(preds.argmax(-1), labels.to(self.device))}
        
    def processEvalIteration(self, model, _curr_batch):
        graphs, labels = _curr_batch
        preds = model(graphs.to(self.device),
                      graphs.ndata['feat'].to(self.device) if 'feat' in graphs.ndata else None,
                      edge_attr = graphs.edata['feat'].to(self.device) if 'feat' in graphs.edata else None,
                      edge_weight = graphs.edata['weight'].to(self.device) if 'weight' in graphs.edata else None)
        loss = self.loss_fn(preds, labels.to(self.device))
        return torch.argmax(preds, dim=-1), {'_num_items': preds.shape[0], 'loss': loss.item(), 'acc': self.eval_fn(preds.argmax(-1), labels.to(self.device))}
    
    def _processTrainingLogs(self, task_id, epoch_cnt, val_metric_result, train_stats, val_stats):
        print('task_id:', task_id, f'Epoch #{epoch_cnt}:', 'train_acc:', round(train_stats['acc'], 4), 'val_acc:', round(val_metric_result, 4), 'train_loss:', round(train_stats['loss'], 4), 'val_loss:', round(val_stats['loss'], 4))
        
    def _processBeforeTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        curr_training_states['scheduler'] = self.scheduler_fn(curr_optimizer)
        curr_training_states['best_val_acc'] = -1.
        curr_training_states['best_val_loss'] = 1e10
        
        curr_model.observe_labels(torch.LongTensor([curr_dataset['train'][i][1] for i in range(len(curr_dataset['train']))]))
        self._reset_optimizer(curr_optimizer)