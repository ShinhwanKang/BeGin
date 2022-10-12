"""
    blib
"""


class BaseLinkIncrementalBenchmark(BaseIncrementalBenchmark):
    """
    Base framework under link-level problems

    :param int model: ...
    :param int scenario: ...
    :param int optimizer_fn: ...
    :param int loss_fn: ...
    :param int,optional device: ...
    :param int,optional kwargs: ...
    """
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        """
        :returns int: a+b
        """
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.scheduler_fn = kwargs['scheduler_fn']
        
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        self._dataloader_seed_worker = seed_worker
        
    def prepareLoader(self, curr_dataset, curr_training_states):
        """
        :returns int: a+b
        """
        graph = curr_dataset['graph'].clone()
        node_feats = graph.ndata.pop('feat')
        curr_training_states['graph'] = (graph, node_feats)
        
        datasets = {_split: (curr_dataset[_split]['edge'], curr_dataset[_split]['label']) for _split in ['train', 'val', 'test']}
        train_dataset = torch.utils.data.TensorDataset(*datasets['train'])
        g_train = torch.Generator()
        g_train.manual_seed(0)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= (64 * 1024) // 2, shuffle=True, drop_last=False, num_workers=4, worker_init_fn=self._dataloader_seed_worker, generator=g_train) 
        return train_loader, [(graph, node_feats, *datasets['val'])], [(graph, node_feats, *datasets['test'])]
    
    def processTrainIteration(self, model, optimizer, _curr_batch, training_states):
        graph, feats = map(lambda x: x.to(self.device), training_states['graph'])
        edges, labels = map(lambda x: x.to(self.device), _curr_batch)
        optimizer.zero_grad()
        srcs, dsts = edges[:, 0], edges[:, 1]
        neg_dsts = torch.randint(low=0, high=graph.num_nodes(), size=(srcs.shape[0],)).to(self.device)
        preds = model(graph, feats,
                      srcs.repeat(2), torch.cat((edges[:, 1], neg_dsts), dim=0)).squeeze(-1)
        labels = labels.to(self.device)
        loss = self.loss_fn(preds, torch.cat((labels, torch.zeros_like(labels)), dim=0))
        loss.backward()
        optimizer.step()
        return {'_num_items': preds.shape[0], 'loss': loss.item()}
        
    def processEvalIteration(self, model, _curr_batch):
        graph, feats, edges, _ = map(lambda x: x.to(self.device), _curr_batch)
        srcs, dsts = edges[:, 0], edges[:, 1]
        preds = model(graph, feats, srcs, dsts).squeeze(-1)
        return preds, {}
    
    def _processAfterEachIteration(self, curr_model, curr_optimizer, curr_training_states, curr_iter_results):
        scheduler = curr_training_states['scheduler']
        val_score = curr_iter_results['val_metric']
        if val_score > curr_training_states['best_val_score']:
            curr_training_states['best_val_score'] = val_score
            curr_training_states['best_weights'] = copy.deepcopy(curr_model.state_dict())
        scheduler.step(val_score)
        
        if -1e-9 < (curr_optimizer.param_groups[0]['lr'] - scheduler.min_lrs[0]) < 1e-9:
            return False
        return True
    
    def _processTrainingLogs(self, task_id, epoch_cnt, val_metric_result, train_stats, val_stats):
        if epoch_cnt % 1 == 0: print('task_id:', task_id, f'Epoch #{epoch_cnt}:', 'train_loss:', round(train_stats['loss'], 4), 'val_hits:', round(val_metric_result, 4))
        
    def _processBeforeTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        curr_training_states['scheduler'] = self.scheduler_fn(curr_optimizer)
        curr_training_states['best_val_score'] = -1.
        curr_model.observe_labels(torch.LongTensor([0]))
        self._reset_optimizer(curr_optimizer)