import torch.optim as optim
import torch
from pytorch_lightning.callbacks import Callback

class PlateauScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr, metricTracker):
        self.initial_lr = lr
        self.min_lr = 0.000001
        self.step_size = 4000
        self.stepCount = 0
        self.warmup_step = 0#5
        self.metricTracker = metricTracker
        self.plateauLR = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.9, min_lr=self.min_lr)
        super().__init__(optimizer, -1, False)

    def get_lr(self):
        return self.adjust_learning_rate(self.optimizer)

    def adjust_learning_rate(self, optimizer):
        if self.stepCount < self.warmup_step and self.metricTracker.batchStep % self.step_size == 0:
            self.stepCount += 1
            """Decay the learning rate with half-cycle cosine after warmup"""
            lr =  self.initial_lr * self.stepCount / self.warmup_step
            for param_group in optimizer.param_groups:
                if "lr_scale" in param_group:
                    param_group["lr"] = lr * param_group["lr_scale"]
                else:
                    param_group["lr"] = lr
            return [lr for _ in self.optimizer.param_groups]
        elif self.stepCount >= self.warmup_step:
            self.plateauLR.step(self.metricTracker.loss)
            


class MetricTracker(Callback):

  def __init__(self):
    self.collection = []
    self.loss = 0
    self.batchStep = 0

  #def on_train_batch_end(trainer, module, outputs, batch, batch_idx, something):
  #  self.loss = outputs['loss']
  #  self.collection.append(self.loss)
  #  self.batchStep += 1

  #def on_train_epoch_end(trainer, module):
  #  elogs = trainer.logged_metrics
  #  self.collection.append(elogs)