import os 
import sys
import datetime 
from copy import deepcopy
import numpy as np 
import pandas as pd 
from argparse import Namespace 

class BaseCallBack:
    def __init__(self):
        pass
    
    def on_training_start(self, model):
        pass
        
    def on_training_epoch_end(self, model):
        pass
        
    def on_validation_epoch_end(self, model):
        pass

    def on_fit_end(self, model):
        pass

class VisMetric:
    def __init__(self,figsize = (6,4)):
        self.figsize = (6,4)
        
    def on_training_start(self,model):
        self.metric =  model.monitor.replace('val_','')
        dfhistory = pd.DataFrame(model.history)
        x_bounds = [0, min(10,model.epochs)]
        title = f'best {model.monitor} = ?'
        self.update_graph(model, title=title, x_bounds = x_bounds)
        
    def on_training_epoch_end(self,model):
        pass
    
    def on_validation_epoch_end(self, model):
        dfhistory = pd.DataFrame(model.history)
        n = len(dfhistory)
        x_bounds = [dfhistory['epoch'].min(), min(10+(n//10)*10,model.epochs)]
        title = self.get_title(model)
        self.update_graph(model, title = title,x_bounds = x_bounds)
        
            
    def on_fit_end(self,  model):
        dfhistory = pd.DataFrame(model.history)
        title = self.get_title(model)
        self.update_graph(model, title = title)
        
    def get_title(self,  model):
        dfhistory = pd.DataFrame(model.history)
        arr_scores = dfhistory[model.monitor]
        best_score = np.max(arr_scores) if model.mode=="max" else np.min(arr_scores)
        title = f'best {model.monitor} = {best_score:.4f}'
        return title

    def update_graph(self, model, title=None, x_bounds=None, y_bounds=None):
        import matplotlib.pyplot as plt
        self.plt = plt
        if not hasattr(self, 'graph_fig'):
            self.graph_fig, self.graph_ax = plt.subplots(1, figsize=self.figsize)
            self.graph_out = display(self.graph_ax.figure, display_id=True)
        self.graph_ax.clear()
        
        dfhistory = pd.DataFrame(model.history)
        epochs = dfhistory['epoch'] if 'epoch' in dfhistory.columns else []
        
        m1 = "train_"+self.metric
        if  m1 in dfhistory.columns:
            train_metrics = dfhistory[m1]
            self.graph_ax.plot(epochs,train_metrics,'bo--',label= m1)

        m2 = 'val_'+self.metric
        if m2 in dfhistory.columns:
            val_metrics = dfhistory[m2]
            self.graph_ax.plot(epochs,val_metrics,'ro-',label =m2)

        if self.metric in dfhistory.columns:
            metric_values = dfhistory[self.metric]
            self.graph_ax.plot(epochs, metric_values,'go-', label = self.metric)

        self.graph_ax.set_xlabel("epoch")
        self.graph_ax.set_ylabel(self.metric)  
        if title:
             self.graph_ax.set_title(title)
        if m1 in dfhistory.columns or m2 in dfhistory.columns or self.metric in dfhistory.columns:
            self.graph_ax.legend(loc='best')

        if x_bounds is not None: self.graph_ax.set_xlim(*x_bounds)
        if y_bounds is not None: self.graph_ax.set_ylim(*y_bounds)
        self.graph_out.update(self.graph_ax.figure)
        self.plt.close()
        

class VisDisplay:
    def __init__(self,display_fn,model=None,init_display=True):
        from ipywidgets import Output 
        self.display_fn = display_fn
        self.init_display = init_display
        self.out = Output()
        
        if self.init_display:
            display(self.out)
            with self.out:
                self.display_fn(model)
        
    def on_training_start(self,model):
        if not self.init_display:
            display(self.out)

    def on_training_epoch_end(self,model):
        pass
    
    def on_validation_epoch_end(self, model):
        self.out.clear_output()
        with self.out:
            self.display_fn(model)
           
    def on_fit_end(self,  model):
        pass
        
class EpochCheckpoint:
    def __init__(self, ckpt_dir="weights", save_freq=1):
        self.ckpt_dir = ckpt_dir
        self.save_freq = save_freq
        self.ckpt_idx = 0
        
    def on_training_start(self, model):
        os.makedirs(self.ckpt_dir, exist_ok=True)
        
    def on_training_epoch_end(self, model):
        pass
        
    def on_validation_epoch_end(self, model):
        weights_name = "checkpoint.pth"
        epoch = model.history['epoch'][-1]
        if epoch > 0 and epoch % self.save_freq == 0:
            ckpt_path = os.path.join(self.ckpt_dir, f'{epoch:0>4d}_of_{model.epochs:0>4d}_{weights_name}')
            net_dict = model.accelerator.get_state_dict(model.network)
            model.accelerator.save(net_dict, ckpt_path)

    def on_fit_end(self, model):
        pass

class EpochTrainLog:
    def __init__(self, save_dir="log", log_name="train_log", save_freq=1, rewrite=True):
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.ckpt_idx = 0
        self.log_path = os.path.join(self.save_dir, f'{log_name}.txt')

        self.rewrite = rewrite
        if rewrite and os.path.exists(self.log_path):
            os.remove(self.log_path)
        
    def on_training_start(self, model):
        os.makedirs(self.save_dir, exist_ok=True)
        
    def on_training_epoch_end(self, model):
        pass
            
    def on_validation_epoch_end(self, model):
        epoch = model.history['epoch'][-1]
        lr = model.history['lr'][-1]

        if epoch > 0 and epoch % self.save_freq == 0:
            nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            metrics_str = "[{}, {:4d}, {:7.5f}] ".format(nowtime, epoch, lr)
            
            for _key, _value in model.history.items():
                if "train" in _key:
                    value = _value[-1]
                    metrics_str += f"{_key}={value:6.4f}, "
            self.save_to_txt(self.log_path, metrics_str + "\n", "a")

    def on_fit_end(self, model):
        pass

    def save_to_txt(self, filename, string, mode="a"):
        with open(filename, mode) as f:
            f.write(string)

class EpochEvalLog(EpochTrainLog):
    def __init__(self, save_dir="log", log_name="val_log", save_freq=1, rewrite=True):
        super().__init__(save_dir, log_name, save_freq, rewrite)
            
    def on_validation_epoch_end(self, model):
        epoch = model.history['epoch'][-1]
        lr = model.history['lr'][-1]

        if epoch > 0 and epoch % self.save_freq == 0:
            nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            metrics_str = "[{}, {:4d}, {:7.5f}] ".format(nowtime, epoch, lr)
            
            for _key, _value in model.history.items():
                if "val" in _key:
                    value = _value[-1]
                    metrics_str += f"{_key}={value:6.4f}, "
            self.save_to_txt(self.log_path, metrics_str + "\n", "a")

class StepTrainLog:
    def __init__(self, save_dir="log", log_name="step_train_log", rewrite=True):
        self.save_dir = save_dir
        self.ckpt_idx = 0
        self.log_path = os.path.join(self.save_dir, f'{log_name}.txt')

        self.rewrite = rewrite
        if rewrite and os.path.exists(self.log_path):
            os.remove(self.log_path)
        
    def on_training_start(self, model):
        os.makedirs(self.save_dir, exist_ok=True)
        
    def on_training_epoch_end(self, model):
        pass
            
    def on_validation_epoch_end(self, model):
        epoch = model.history['epoch'][-1]
        lr = model.history['lr'][-1]

        step_metrics = model.train_step_metrics
        
        epoch_length = len(step_metrics["lr"])
        for step_id in range(epoch_length):
            step = step_id + epoch_length * (epoch - 1)
            metrics_str = f"[{epoch}, {step}] "
            for _key, _value in step_metrics.items():
                if "train" in _key:
                    value = _value[step_id]
                    metrics_str += f"{_key}={value:6.4f}, "
            self.save_to_txt(self.log_path, metrics_str + "\n", "a")

    def on_fit_end(self, model):
        pass

    def save_to_txt(self, filename, string, mode="a"):
        with open(filename, mode) as f:
            f.write(string)