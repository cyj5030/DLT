import os
import torch
import torch.nn as nn

import optuna
from optuna import Trial

import inspect
from typing import Any, List, Dict
from functools import reduce
from operator import mul

from torch.utils.data import Dataset, DataLoader
from accelerate.utils import set_seed

from .model import Model
from .hps_callback import Pruning
from .utils import pkl_save, pkl_load, colorful

Optimizer = {
    "Adadelta": torch.optim.Adadelta,
    "Adagrad": torch.optim.Adagrad,
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "SparseAdam": torch.optim.SparseAdam,
    "Adamax": torch.optim.Adamax,
    "ASGD": torch.optim.ASGD,
    "SGD": torch.optim.SGD,
    "RAdam": torch.optim.RAdam,
    "Rprop": torch.optim.Rprop,
    "RMSprop": torch.optim.RMSprop,
    "NAdam": torch.optim.NAdam,
}

LrScheduler = {
    "LambdaLR": torch.optim.lr_scheduler.LambdaLR,
    "MultiplicativeLR": torch.optim.lr_scheduler.MultiplicativeLR,
    "StepLR": torch.optim.lr_scheduler.StepLR,
    "MultiStepLR": torch.optim.lr_scheduler.MultiStepLR,
    "ConstantLR": torch.optim.lr_scheduler.ConstantLR,
    "LinearLR": torch.optim.lr_scheduler.LinearLR,
    "ExponentialLR": torch.optim.lr_scheduler.ExponentialLR,
    "SequentialLR": torch.optim.lr_scheduler.SequentialLR,
    "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
    "ChainedScheduler": torch.optim.lr_scheduler.ChainedScheduler,
    "CyclicLR": torch.optim.lr_scheduler.CyclicLR,
    "CosineAnnealingWarmRestarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    "OneCycleLR": torch.optim.lr_scheduler.OneCycleLR,
}

Pruner = {
    "HyperbandPruner": optuna.pruners.HyperbandPruner,
    "MedianPruner": optuna.pruners.MedianPruner,
    "NopPruner": optuna.pruners.NopPruner,
    "PatientPruner": optuna.pruners.PatientPruner,
    "PercentilePruner": optuna.pruners.PercentilePruner,
    "SuccessiveHalvingPruner": optuna.pruners.SuccessiveHalvingPruner,
    "ThresholdPruner": optuna.pruners.ThresholdPruner,
}

def check_keys(keys, query):
    for _key in keys:
        assert _key in query, "key is not in query"

def check_obj(obj, params_space):
    kwarg = inspect.signature(obj)
    query = list(kwarg.parameters.keys())
    keys = list(params_space.keys())
    check_keys(keys, query)

def check_objs(objs, params_spaces):
    for obj, params_space in zip(objs, params_spaces):
        check_obj(obj, params_space)

class HpsOp:
    def __init__(self) -> None:
        pass
        
class HpsAdd(HpsOp):
    def __init__(self, x, key) -> None:
        self.x = x
        self.key = key

    def __call__(self, obj) -> Any:
        return obj[self.key] + self.x

class HpsMul(HpsOp):
    def __init__(self, x, key) -> None:
        self.x = x
        self.key = key

    def __call__(self, obj) -> Any:
        return obj[self.key] * self.x


def get_trial_length(ps):
    n = 1
    if ps:
        n = reduce(mul, [len(_value) if _value is not None else 1 for _value in ps.values()])
    return n

class ModuleSpace:
    def __init__(self, obj: nn.Module, init_ps) -> None:
        self.obj = obj
        self.init_ps = init_ps

    def create(self, trial: Trial):
        network_params = {}
        for _key, _value in self.init_ps.items():
            network_params[_key] = trial.suggest_categorical('module.' + _key, _value)
        network = self.obj(**network_params)
        return network
    
    @property
    def n_trials(self):
        n = get_trial_length(self.init_ps)
        return max(1, n)

class DataSpace:
    def __init__(self, obj: Dataset, dataset_ps, dataloader_ps) -> None:
        self.obj = obj
        self.dataset_ps = dataset_ps
        self.dataloader_ps = dataloader_ps

    def create(self, trial: Trial):
        dataset_params = {}
        for _key, _value in self.dataset_ps.items():
            dataset_params[_key] = trial.suggest_categorical("dataset." + _key, _value)
        dataset = self.obj(**dataset_params)

        dataloader_params = {}
        for _key, _value in self.dataloader_ps.items():
            dataloader_params[_key] = trial.suggest_categorical("dataloader." + _key, _value)
        dataloader = DataLoader(dataset, **dataloader_params)
        return dataloader

    @property
    def n_trials(self):
        n1 = get_trial_length(self.dataset_ps)
        n2 = get_trial_length(self.dataloader_ps)
        return max(1, n1 * n2)
    
class ValDataSpace:
    def __init__(self, obj: Dataset, dataset_ps, dataloader_ps) -> None:
        self.obj = obj
        self.dataset_ps = dataset_ps
        self.dataloader_ps = dataloader_ps

    def create(self, trial: Trial, train_loader):
        dataset_params = {}
        for _key, _value in self.dataset_ps.items():
            if _value is None:
                value = inspect.signature(train_loader.dataset.__init__).parameters.get(_key).default
            else:
                value = trial.suggest_categorical("val_dataset." + _key, _value)
            dataset_params[_key] = value        
        dataset = self.obj(**dataset_params)

        dataloader_params = {}
        for _key, _value in self.dataloader_ps.items():
            dataloader_params[_key] = trial.suggest_categorical("val_dataloader." + _key, _value)
        dataloader = DataLoader(dataset, **dataloader_params)
        return dataloader
    
    @property
    def n_trials(self):
        n1 = get_trial_length(self.dataset_ps)
        n2 = get_trial_length(self.dataloader_ps)
        return max(1, n1 * n2)
    
class OptimSpace:
    def __init__(self, optim, optim_ps) -> None:
        self.optim = optim
        self.optim_ps = optim_ps

    def create(self, trial: Trial, network_params):
        optim_params = {}
        for _key, _value in self.optim_ps.items():
            optim_params[_key] = trial.suggest_categorical("optimizer." + _key, _value)
        optimizer_fn = Optimizer[trial.suggest_categorical('optimizer', self.optim)]
        optimizer = optimizer_fn(network_params, **optim_params)
        return optimizer

    @property
    def n_trials(self):
        n1 = get_trial_length(self.optim_ps)
        n2 = len(self.optim)
        return max(1, n1 * n2)
    
class LrSchedulerSpace:
    def __init__(self, lr_scheduler, lr_scheduler_ps) -> None:
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_ps = lr_scheduler_ps

    def create(self, trial: Trial, optim, dataloader, epoch):
        defaults = {}
        defaults["lr"] = optim.defaults["lr"]
        defaults["total_steps"] = int(epoch * len(dataloader))
        defaults["epoch"] = epoch
        defaults["steps_per_epoch"] = len(dataloader)

        lr_params = {}
        for _key, _value in self.lr_scheduler_ps.items():
            value = trial.suggest_categorical("lr_scheduler." + _key, _value)
            if isinstance(value, HpsOp):
                lr_params[_key] = value(defaults)
            else:
                lr_params[_key] = value
        lr_fn = LrScheduler[trial.suggest_categorical('lr_scheduler', self.lr_scheduler)]
        scheduler = lr_fn(optim, **lr_params)
        return scheduler
    
    @property
    def n_trials(self):
        n1 = get_trial_length(self.lr_scheduler_ps)
        n2 = len(self.lr_scheduler)
        return max(1, n1 * n2)
    
class HyperParamSearch(nn.Module):
    
    def __init__(self,
                network: ModuleSpace, 
                loss_fn: ModuleSpace,
                optimizer: OptimSpace,
                train_dataset: DataSpace, 
                val_dataset: ValDataSpace,
                epoch_space: Dict[str, Any],
                lr_scheduler: LrSchedulerSpace = None,
                metrics_dict: Dict[str, Any] = None,
                seed_space=False,
                log_folder='.'
                ) -> None:
        super().__init__()
        self.study = None

        self.network = network
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.epoch_space = epoch_space
        self.metrics_dict = metrics_dict

        self.seed_space = seed_space

        os.makedirs(log_folder, exist_ok=True)
        self.log_folder = log_folder

    def objective(self, trial: Trial, callback_fn, monitor):
        epoch = trial.suggest_categorical("epoch", self.epoch_space)
        if self.seed_space is not None:
            set_seed(trial.suggest_categorical("seed", self.seed_space))

        train_loader = self.train_dataset.create(trial)
        val_loader = self.val_dataset.create(trial, train_loader)

        network = self.network.create(trial)
        loss = self.loss_fn.create(trial)

        optimizer = self.optimizer.create(trial, network.parameters())
        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler.create(trial, optimizer, train_loader, epoch)
        
        for callback_obj in callback_fn:
            callback_obj.set_trial(trial)

        custom_model = Model(network, loss, optimizer, metrics_dict=self.metrics_dict, lr_scheduler=lr_scheduler)
        history = custom_model.fit(train_loader, 
                        val_loader, 
                        epochs=epoch, 
                        ckpt_path=None, 
                        callbacks=callback_fn,
                        early_stopping=False,
                        # patience=epoch, 
                        # monitor="val_accuracy",
                        # mode="max",
                        quiet=True,
                        )
        metric = history[monitor][-1]
        return metric

    def fit(self, study_name, callback_fn=None, monitor="val_loss", direction="minimize", n_trials=100, pruner=None):
        # direction: ['minimize', 'maximize']
        max_n_trials = self.get_total_trial()
        if n_trials > max_n_trials:
            n_trials = max_n_trials
            print(colorful(f"<<<<<<<<<<<<<<<<< reset ntrial to {max_n_trials} >>>>>>>>>>>>>>>>>>"))

        if pruner is not None:
            self.study = optuna.create_study(study_name=study_name, pruner=pruner, direction=direction)
        else:
            self.study = optuna.create_study(study_name=study_name, direction=direction)

        objective_fn = lambda trial: self.objective(trial, callback_fn, monitor)
        self.study.optimize(objective_fn, n_trials=n_trials)
        pkl_save(self.study, os.path.join(self.log_folder, study_name, f"{study_name}.pkl"))

    def get_total_trial(self):
        n = self.network.n_trials * self.loss_fn.n_trials * self.optimizer.n_trials * self.lr_scheduler.n_trials * \
            self.train_dataset.n_trials * self.val_dataset.n_trials * len(self.epoch_space) * len(self.seed_space)
        return n
    
    def print_study(self):
        print(self.study.best_params)


def load_from_study(study_name):
    study = pkl_load(study_name)
    return study