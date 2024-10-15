import torch.nn as nn
import torch.optim as optim
import numpy as np
import cuml
import cupy as cp
from typing import Any
from functools import partial
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import _LRScheduler


class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, warm_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = warm_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        # 父类属性，会在 super().__init__(optimizer) 中初始化
        # self.last_epoch = None
        # self.base_lrs = None
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]


    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)
        

def build_optimizer(optim_args, optim_sch_arg, params) -> tuple[Optimizer, LRScheduler | None]:
    filter_fn = filter(lambda p : p.requires_grad, params) # 只选择需要更新的参数
    if optim_args.type == 'adam':
        optimizer = optim.Adam(filter_fn, lr=optim_args.lr, weight_decay=optim_args.weight_decay)
    elif optim_args.type == 'adamw':
        optimizer = optim.AdamW(filter_fn, lr=optim_args.lr, weight_decay=optim_args.weight_decay)
    elif optim_args.type == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=optim_args.lr, momentum=optim_args.momentum, weight_decay=optim_args.weight_decay)
    elif optim_args.type == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=optim_args.lr, weight_decay=optim_args.weight_decay)
    elif optim_args.type == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=optim_args.lr, weight_decay=optim_args.weight_decay)
    else:
        raise NotImplementedError(f"Unknown optimizer type {optim_args.type}")
    if optim_sch_arg is None:
        return optimizer, None 
    elif optim_sch_arg.type == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=optim_sch_arg.decay_step, gamma=optim_sch_arg.decay_rate)
    elif optim_sch_arg.type == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=optim_sch_arg.opt_restart)
    else:
        raise NotImplementedError(f"Unknown scheduler type {optim_sch_arg}")

    if optim_sch_arg.warmup is not None:
        scheduler = GradualWarmupScheduler(optimizer, 
                                           multiplier=optim_sch_arg.warmup.multiplier, 
                                           warm_epoch=optim_sch_arg.warmup.epoch, 
                                           after_scheduler=scheduler)

    return optimizer, scheduler 


class Criterion(object):
    metrics = {
        "acc": accuracy_score,
        "f1": f1_score,
        "auc": roc_auc_score
    }

    def __init__(self, name: str, **params) -> None:
        self.name = name
        self.params = params
        self.metric = partial(self.metrics[name], **params)

    def __call__(self, y_true: Tensor, y_pred: Tensor) -> float:

        return self.metric(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
    
    def __repr__(self) -> str:
        prefix = f"{self.params['average']}_" if self.params.get("average") else ""
        return prefix + self.name


class EMAHelper(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(
                inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        # module_copy = copy.deepcopy(module)
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict
