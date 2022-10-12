class BaseIncrementalBenchmark(BaseContinualFramework):
    """
    Base framework under node-level problems22

    :param int model: ...
    :param int scenario: ...
    :param int optimizer_fn: ...
    :param int loss_fn: ...
    :param int,optional device: ...
    :param int,optional kwargs: ...
    """
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.scheduler_fn = kwargs['scheduler_fn']
        
    def prepareLoader(self, curr_dataset, curr_training_states):
        return [(curr_dataset, curr_dataset.ndata['train_mask'])], [(curr_dataset, curr_dataset.ndata['val_mask'])], [(curr_dataset, curr_dataset.ndata['test_mask'])]
    
    def processTrainIteration(self, model, optimizer, _curr_batch, training_states):
        curr_batch, mask = _curr_batch
        optimizer.zero_grad()
        preds = model(curr_batch.to(self.device), curr_batch.ndata['feat'].to(self.device))[mask]
        loss = self.loss_fn(preds, curr_batch.ndata['label'][mask].to(self.device))
        loss.backward()
        optimizer.step()
        return {'loss': loss.item(), 'acc': self.eval_fn(preds.argmax(-1), curr_batch.ndata['label'][mask].to(self.device))}
        
    def processEvalIteration(self, model, _curr_batch):
        curr_batch, mask = _curr_batch
        preds = model(curr_batch.to(self.device), curr_batch.ndata['feat'].to(self.device))[mask]
        loss = self.loss_fn(preds, curr_batch.ndata['label'][mask].to(self.device))
        return torch.argmax(preds, dim=-1), {'loss': loss.item()}
    
    def _processTrainingLogs(self, task_id, epoch_cnt, val_metric_result, train_stats, val_stats):
        if epoch_cnt % 10 == 0: print('task_id:', task_id, f'Epoch #{epoch_cnt}:', 'train_acc:', round(train_stats['acc'], 4), 'val_acc:', round(val_metric_result, 4), 'train_loss:', round(train_stats['loss'], 4), 'val_loss:', round(val_stats['loss'], 4))
        pass
    
    def _initTrainingStates(self, scenario, model, optimizer):
        return {}
    
    def _processBeforeTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        curr_training_states['scheduler'] = self.scheduler_fn(curr_optimizer)
        curr_training_states['best_val_acc'] = -1.
        curr_training_states['best_val_loss'] = 1e10
        curr_model.observe_labels(curr_dataset.ndata['label'][curr_dataset.ndata['train_mask']])
        self._reset_optimizer(curr_optimizer)
        
    def _processAfterEachIteration(self, curr_model, curr_optimizer, curr_training_states, curr_iter_results):
        scheduler = curr_training_states['scheduler']
        val_acc = curr_iter_results['val_metric']
        val_loss = curr_iter_results['val_stats']['loss']
        if val_acc > curr_training_states['best_val_acc']:
            curr_training_states['best_val_acc'] = val_acc
        # print((curr_training_states['best_val_loss'] - val_loss) / curr_training_states['best_val_loss'], curr_training_states['best_val_loss'], val_loss)
        if val_loss < curr_training_states['best_val_loss']:
            curr_training_states['best_val_loss'] = val_loss
            curr_training_states['best_weights'] = copy.deepcopy(curr_model.state_dict())
        scheduler.step(val_loss)
        
        # scheduler.step(val_acc)
        if -1e-9 < (curr_optimizer.param_groups[0]['lr'] - scheduler.min_lrs[0]) < 1e-9:
            # earlystopping!
            return False
        return True
    
    def _processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        curr_model.load_state_dict(curr_training_states['best_weights'])
    
    def computeFinalMetrics(self, epoch_per_task=1):
        results = self.fit(epoch_per_task=epoch_per_task)
        
        with open(f'{self.result_path}/{self.save_file_name}.pkl', 'wb') as f:
            pickle.dump({k: v.detach().cpu().numpy() for k, v in results.items()}, f)
        if self.full_mode:
            init_acc, accum_acc_mat, base_acc_mat, algo_acc_mat = map(lambda x: results[x].detach().cpu().numpy(), ('init_test', 'accum_test', 'base_test', 'exp_test'))
        else:
            init_acc, algo_acc_mat = map(lambda x: results[x].detach().cpu().numpy(), ('init_test', 'exp_test'))
            
        print('init_acc:', init_acc)
        print('algo_acc_mat:', algo_acc_mat)
        print('abs_avg_precision_last:', round(algo_acc_mat[self.num_tasks - 1][:-1].sum() / self.num_tasks, 4))
        print('abs_avg_precision_diag:', round(algo_acc_mat[np.arange(self.num_tasks), np.arange(self.num_tasks)].sum() / self.num_tasks, 4))
        print('abs_avg_forgetting_last:', round((algo_acc_mat[np.arange(self.num_tasks), np.arange(self.num_tasks)] - algo_acc_mat[self.num_tasks - 1, :self.num_tasks]).sum() / (self.num_tasks - 1), 4))
        if self.full_mode:
            print('base_acc_mat:', base_acc_mat)
            print('joint_acc_mat:', accum_acc_mat)
            print('abs_avg_intransigence_base:', round((base_acc_mat - algo_acc_mat)[np.arange(self.num_tasks), np.arange(self.num_tasks)].mean(), 4))
            print('abs_avg_intransigence_joint:', round((accum_acc_mat - algo_acc_mat)[np.arange(self.num_tasks), np.arange(self.num_tasks)].mean(), 4))
            print('rel_upstream_transfer_diag:', round((((algo_acc_mat - base_acc_mat)[np.arange(self.num_tasks), np.arange(self.num_tasks)]) / (base_acc_mat[np.arange(self.num_tasks), np.arange(self.num_tasks)] - init_acc[:self.num_tasks])).mean(), 4))
          