import torch.optim as optim
import math

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epoch, lr, epochs):
        self.warmup_epoch = warmup_epoch + 1
        self.total_epochs = epochs
        self.initial_lr = lr
        super().__init__(optimizer, -1, False)


    def get_lr(self):
        return self.adjust_learning_rate(self.optimizer, self.last_epoch)

    def adjust_learning_rate(self, optimizer, epoch):
        min_lr = 0.000001
        epoch = epoch + 1
        """Decay the learning rate with half-cycle cosine after warmup"""
        if epoch < self.warmup_epoch:
            lr =  self.initial_lr * epoch / self.warmup_epoch
        else:
            lr = min_lr + (self.initial_lr - min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - self.warmup_epoch) / (self.total_epochs - self.warmup_epoch)))
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
        return [lr for _ in self.optimizer.param_groups]