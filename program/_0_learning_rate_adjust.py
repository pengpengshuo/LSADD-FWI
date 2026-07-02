from torch.optim.lr_scheduler import _LRScheduler
import math

class CosineWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=0.0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        #warm up
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            #Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return [self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
                    for base_lr in self.base_lrs]



class WarmUp(object):
    def __init__(self,optimizer,epoch,idx=0,action=True):
        self.optimizer=optimizer
        self.epoch=epoch
        self.__idx=idx
        self.action=action
        self.base_lr = self.optimizer.param_groups[-1]['lr']
    def get_lr(self):
        if self.__idx+1 <=self.epoch:
            base_lr=self.base_lr*((self.__idx+1)/self.epoch)
            self.__idx=self.__idx+1
        else:
            base_lr=self.base_lr
        self.optimizer.param_groups[-1]['lr'] = base_lr
    def step(self):
        if self.action is True:
            self.get_lr()

