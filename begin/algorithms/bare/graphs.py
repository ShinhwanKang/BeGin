import sys
from begin.trainers.graphs import GCTrainer

class GCTaskILBareTrainer(GCTrainer):
    def inference(self, model, _curr_batch, training_states):
        graphs, labels, masks = _curr_batch
        preds = model(graphs.to(self.device),
                      graphs.ndata['feat'].to(self.device) if 'feat' in graphs.ndata else None,
                      edge_attr = graphs.edata['feat'].to(self.device) if 'feat' in graphs.edata else None,
                      edge_weight = graphs.edata['weight'].to(self.device) if 'weight' in graphs.edata else None,
                      task_masks = masks)
        loss = self.loss_fn(preds, labels.to(self.device))
        return {'preds': preds, 'loss': loss}

class GCClassILBareTrainer(GCTrainer):
    pass

class GCDomainILBareTrainer(GCTrainer):
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
        return {'_num_items': preds.shape[0], 'loss': loss.item(), 'acc': self.eval_fn(preds, labels.to(self.device))}
        
    def processEvalIteration(self, model, _curr_batch):
        graphs, labels = _curr_batch
        preds = model(graphs.to(self.device),
                      graphs.ndata['feat'].to(self.device) if 'feat' in graphs.ndata else None,
                      edge_attr = graphs.edata['feat'].to(self.device) if 'feat' in graphs.edata else None,
                      edge_weight = graphs.edata['weight'].to(self.device) if 'weight' in graphs.edata else None)
        loss = self.loss_fn(preds, labels.to(self.device))
        return preds, {'_num_items': preds.shape[0], 'loss': loss.item(), 'acc': self.eval_fn(preds, labels.to(self.device))}
        
    def processBeforeTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        curr_training_states['scheduler'] = self.scheduler_fn(curr_optimizer)
        curr_training_states['best_val_acc'] = -1.
        curr_training_states['best_val_loss'] = 1e10
        curr_model.observe_labels(torch.LongTensor([0]))
        self._reset_optimizer(curr_optimizer)
        
class GCTimeILBareTrainer(GCClassILBareTrainer):
    pass