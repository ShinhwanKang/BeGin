import sys
from begin.trainers.links import LCTrainer, LPTrainer

class LCTaskILBareTrainer(LCTrainer):
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
    
class LCClassILBareTrainer(LCTrainer):
    pass

class LCTimeILBareTrainer(LCTrainer):
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
        results['loss'].backward()
        optimizer.step()
        return {'loss': results['loss'].item(), 'acc': self.eval_fn(results['preds'], _curr_batch[-1].to(self.device))}

class LPTimeILBareTrainer(LPTrainer):
    pass

class LPDomainILBareTrainer(LPTrainer):
    pass