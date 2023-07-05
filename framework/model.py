import sys
import datetime
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import Any, List, Dict, Callable

import os
from accelerate import Accelerator
from accelerate import notebook_launcher
from accelerate.utils import write_basic_config
from accelerate.commands.config.config_args import default_config_file
if not os.path.exists(default_config_file):
    write_basic_config() # Write a config file

from .utils import colorful, is_jupyter

class StepRunner:
    def __init__(self, network: nn.Module, 
                 loss_fn: nn.Module, 
                 accelerator: Accelerator, 
                 stage = "train", metrics_dict = None, 
                 optimizer = None, lr_scheduler = None
                 ):
        self.network = network
        self.loss_fn = loss_fn
        self.metrics_dict = metrics_dict
        self.stage = stage
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        if self.stage == 'train':
            self.network.train() 
        else:
            self.network.eval()
    
    def __call__(self, batch):
        features, labels = batch 
        
        # loss
        with self.accelerator.autocast():
            preds = self.network(features)
            loss = self.loss_fn(preds, labels)

        # backward()
        if self.optimizer is not None and self.stage == "train":
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.network.parameters(), 1.0)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
            
        all_preds = self.accelerator.gather(preds)
        all_labels = self.accelerator.gather(labels)
        all_loss = self.accelerator.gather(loss).sum()
        
        # losses (plain metrics)
        step_losses = {self.stage + "_loss": all_loss.item()}
        
        # metrics (state metrics)
        step_metrics = {}
        if self.metrics_dict is not None:
            step_metrics.update(
                {self.stage + "_" + name: metric_fn(all_preds, all_labels) for name, metric_fn in self.metrics_dict.items()}
            )
        
        if self.stage == "train":
            if self.optimizer is not None:
                step_metrics['lr'] = self.optimizer.state_dict()['param_groups'][0]['lr']
            else:
                step_metrics['lr'] = 0.0

        return step_losses, step_metrics

class EpochRunner:
    def __init__(self, steprunner, quiet=False):
        self.steprunner = steprunner
        self.stage = steprunner.stage
        self.accelerator = steprunner.accelerator
        self.network = steprunner.network
        self.quiet = quiet

        self.step_losses = {}
        self.step_metrics = {}
        
    def __call__(self, dataloader):
        n = dataloader.size  if hasattr(dataloader, 'size') else len(dataloader)
        loop = tqdm(enumerate(dataloader, start=1), 
                    total=n,
                    file=sys.stdout,
                    disable=not self.accelerator.is_local_main_process or self.quiet,
                    ncols=100
                   )
        epoch_losses = {}
        epoch_metrics = {}
        for step, batch in loop: 
            with self.accelerator.accumulate(self.network):

                step_losses, step_metrics = self.steprunner(batch)

                for k, v in step_losses.items():
                    epoch_losses[k] = epoch_losses.get(k, 0.0) + v
                    self.step_losses[k] = self.step_losses.get(k, []) + [v]

                if self.steprunner.metrics_dict is not None:
                    for k, v in step_metrics.items():
                        epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v
                        self.step_metrics[k] = self.step_metrics.get(k, []) + [v]

                step_log = dict(step_losses, **step_metrics)
                if step < n:
                    loop.set_postfix(**step_log)

                elif step == n:
                    epoch_metrics = {k: v / step for k, v in epoch_metrics.items()}
                    epoch_losses = {k: v / step for k, v in epoch_losses.items()}
                    epoch_log = dict(epoch_losses, **epoch_metrics)
                    loop.set_postfix(**epoch_log)
                    
                else:
                    break

        return epoch_log

class Model(nn.Module):
    
    StepRunner = StepRunner
    EpochRunner = EpochRunner

    def __init__(self, network: nn.Module, 
                 loss_fn: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 metrics_dict: Dict[str, Any] = None,
                 lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None
                ) -> None:
        super().__init__()

        self.network = network
        self.loss_fn = loss_fn
        # self.metrics_dict = None if metrics_dict is None else nn.ModuleDict(metrics_dict)
        self.metrics_dict = metrics_dict
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        
    def load_ckpt(self, ckpt_path='checkpoint.ckpt'):
        self.network.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

    def forward(self, x):
        return self.network.forward(x)
    
    def fit(self, train_data: DataLoader, 
            val_data: DataLoader = None, 
            epochs=10, ckpt_path='checkpoint.ckpt',
            patience=5, monitor="val_loss", mode="min", early_stopping=True, 
            callbacks=None, quiet=False, 
            mixed_precision='no', cpu=False, gradient_accumulation_steps=1):
        
        self.__dict__.update(locals())

        self.accelerator = Accelerator(mixed_precision=mixed_precision,cpu=cpu,
            gradient_accumulation_steps=gradient_accumulation_steps)
        
        device = str(self.accelerator.device)
        if not quiet:
            device_type = '🐌'  if 'cpu' in device else '⚡️'
            self.accelerator.print(
                colorful("<<<<<< " + device_type + " " + device + " is used >>>>>>"))

        self.network, self.loss_fn, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.network, self.loss_fn, self.optimizer, self.lr_scheduler)
        train_dataloader, val_dataloader = self.accelerator.prepare(train_data, val_data)
        # if self.metrics_dict is not None:
        #     self.metrics_dict = self.accelerator.prepare(self.metrics_dict)

        self.history = {}
        callbacks = callbacks if callbacks is not None else []
        self.callbacks = [self.accelerator.prepare(x) for x in callbacks]
        
        if self.accelerator.is_local_main_process:
            for callback_obj in self.callbacks:
                callback_obj.on_training_start(model = self)
                
        # if quiet is None:
        #     if is_jupyter():
        #         quiet = True
        #     else:
        #         quiet = False
        
        # quiet_fn = (lambda epoch: quiet) if isinstance(quiet, bool) else (
        #     (lambda epoch: epoch > quiet) if isinstance(quiet, int) else quiet)
        quiet_fn = (lambda epoch: quiet)

        start_epoch = 1
        for epoch in range(start_epoch, epochs + start_epoch):
            should_quiet = quiet_fn(epoch)
            self.epoch = epoch

            if not should_quiet:
                nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.accelerator.print("\n" + "=========="*4 + f"{nowtime}" + "=========="*4)
                self.accelerator.print("Epoch {0} / {1}".format(epoch, epochs) + "\n")
            
            # 1，train -------------------------------------------------  
            train_step_runner = self.StepRunner(
                    network = self.network,
                    loss_fn = self.loss_fn,
                    accelerator = self.accelerator,
                    stage = "train",
                    metrics_dict = self.metrics_dict,
                    optimizer = self.optimizer,
                    lr_scheduler = self.lr_scheduler,
            )

            train_epoch_runner = self.EpochRunner(train_step_runner, should_quiet)
            train_metrics = {'epoch': epoch}
            train_metrics.update(train_epoch_runner(train_dataloader))

            self.train_step_metrics = train_epoch_runner.step_metrics

            for name, metric in train_metrics.items():
                self.history[name] = self.history.get(name, []) + [metric]

            if self.accelerator.is_local_main_process:
                for callback_obj in self.callbacks:
                    callback_obj.on_training_epoch_end(model = self)

            # 2，validate -------------------------------------------------
            if val_dataloader is not None:
                val_step_runner = self.StepRunner(
                    network = self.network,
                    loss_fn = self.loss_fn,
                    accelerator = self.accelerator,
                    stage = "val",
                    metrics_dict = self.metrics_dict
                )
                val_epoch_runner = self.EpochRunner(val_step_runner, should_quiet)
                with torch.no_grad():
                    val_metrics = val_epoch_runner(val_dataloader)

                for name, metric in val_metrics.items():
                    self.history[name] = self.history.get(name, []) + [metric]
                
            if self.accelerator.is_local_main_process:
                for callback_obj in self.callbacks:
                    callback_obj.on_validation_epoch_end(model = self)

            # 3，early-stopping -------------------------------------------------
            self.accelerator.wait_for_everyone()
            if ckpt_path is not None:
                if early_stopping:
                    arr_scores = self.history[monitor]
                    best_score_idx = np.argmax(arr_scores) if mode == "max" else np.argmin(arr_scores)

                    if best_score_idx == len(arr_scores) - 1:
                        net_dict = self.accelerator.get_state_dict(self.network)
                        self.accelerator.save(net_dict, ckpt_path)
                        if not should_quiet:
                            self.accelerator.print(colorful("<<<<<< reach best {0} : {1:.3f} >>>>>>".format(
                                monitor, arr_scores[best_score_idx])))

                    if len(arr_scores) - best_score_idx > patience:
                        self.accelerator.print(colorful(
                            "<<<<<< {} without improvement in {} epoch,""early stopping >>>>>>"
                        ).format(monitor, patience))
                        break
                else:
                    net_dict = self.accelerator.get_state_dict(self.network)
                    self.accelerator.save(net_dict, ckpt_path)
                
        if self.accelerator.is_local_main_process:   
            for callback_obj in self.callbacks:
                callback_obj.on_fit_end(model = self)
            
            if ckpt_path is not None:
                self.network = self.accelerator.unwrap_model(self.network)
                self.network.cpu()
                self.load_ckpt(ckpt_path)
            
            history = deepcopy(self.history)
            return history
    
    @torch.no_grad()
    def evaluate(self, val_data, quiet=False):
        accelerator = Accelerator() if not hasattr(self,'accelerator') else self.accelerator
        self.network, self.loss_fn, self.metrics_dict = accelerator.prepare(self.network, self.loss_fn, self.metrics_dict)
        val_data = accelerator.prepare(val_data)
        val_step_runner = self.StepRunner(network = self.network, stage="val",
                    loss_fn = self.loss_fn, metrics_dict=deepcopy(self.metrics_dict),
                    accelerator = accelerator)
        val_epoch_runner = self.EpochRunner(val_step_runner, quiet=quiet)
        val_metrics = val_epoch_runner(val_data)
        return val_metrics
    
    def fit_ddp(self, train_data,
            val_data=None, epochs=10, ckpt_path='checkpoint.pt',
            patience=5, monitor="val_loss", mode="min", early_stopping=True,
            callbacks=None, quiet=None, 
            mixed_precision='no', cpu=False, gradient_accumulation_steps=1,
            num_processes=2,
           ):
        # import multiprocessing
        # multiprocessing.set_start_method('spawn')

        args = (train_data, val_data, epochs, ckpt_path, patience, monitor, mode, early_stopping,
            callbacks, quiet, mixed_precision, cpu, gradient_accumulation_steps)
    
        notebook_launcher(self.fit, args, num_processes=num_processes)

@torch.no_grad()
def inference(network, val_data, ckpt_path, mixed_precision='no', cpu=False):
    # accelerator = Accelerator(mixed_precision=mixed_precision, cpu=cpu)
    
    # network = accelerator.prepare(network)
    # val_data = accelerator.prepare(val_data)
    network.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
    network.eval()
    network.cuda()

    outputs = {"features": [], "preds": [], "labels": []}
    for step, batch in tqdm(enumerate(val_data), desc="inference"): 
        # with accelerator.accumulate(network):
        features, labels = batch
        features = {k: v.cuda() for k, v in features.items()}
        labels = {k: v.cuda() for k, v in labels.items()}
        preds = network(features)
        # print(preds["feature"].mean().cpu())
        outputs["preds"].append( {k: v.cpu() for k, v in preds.items()} )
        outputs["labels"].append( {k: v.cpu() for k, v in labels.items()} )
        outputs["features"].append( {k: v.cpu() for k, v in features.items()} )
    return outputs