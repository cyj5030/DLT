import os 
import optuna
import datetime 

class BaseCallBack:
    def __init__(self):
        self.trial = None
    
    def on_training_start(self, model):
        pass
        
    def on_training_epoch_end(self, model):
        pass
        
    def on_validation_epoch_end(self, model):
        pass

    def on_fit_end(self, model):
        pass
    
    def set_trial(self, trial):
        self.trial = trial

class Pruning(BaseCallBack):
    def __init__(self, monitor="val_loss"):
        super().__init__()
        self.monitor = monitor

    def on_validation_epoch_end(self, model):
        epoch = model.history['epoch'][-1]
        self.trial.report(model.history[self.monitor][-1], epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()

class EpochCheckpoint(BaseCallBack):
    def __init__(self, ckpt_dir="weights", save_freq=1):
        super().__init__()
        self.ckpt_dir = ckpt_dir
        self.save_freq = save_freq
        self.ckpt_idx = 0
        self.trial_ckpt_dir = None
        
    def on_training_start(self, model):
        os.makedirs(self.trial_ckpt_dir, exist_ok=True)
        
    def on_validation_epoch_end(self, model):
        weights_name = "checkpoint.pth"
        epoch = model.history['epoch'][-1]
        if epoch > 0 and epoch % self.save_freq == 0:
            ckpt_path = os.path.join(self.trial_ckpt_dir, f'{epoch:0>4d}_of_{model.epochs:0>4d}_{weights_name}')
            net_dict = model.accelerator.get_state_dict(model.network)
            model.accelerator.save(net_dict, ckpt_path)

    def set_trial(self, trial):
        last_dir = os.path.dirname(self.ckpt_dir)
        current_dir = os.path.basename(self.ckpt_dir)
        self.trial_ckpt_dir = os.path.join(last_dir, f"{trial.number:0>3d}", current_dir)
        self.trial = trial

class EpochTrainLog(BaseCallBack):
    def __init__(self, save_dir="log", log_name="train_log", save_freq=1, rewrite=True):
        super().__init__()
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.ckpt_idx = 0
        self.log_name = log_name
        self.log_path = os.path.join(self.save_dir, f'{log_name}.txt')
        self.trial_log_path = None

        self.rewrite = rewrite
        
    def on_training_start(self, model):
        os.makedirs(os.path.dirname(self.trial_log_path), exist_ok=True)
        if self.rewrite and os.path.exists(self.trial_log_path):
            os.remove(self.trial_log_path)
                    
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
            self.save_to_txt(self.trial_log_path, metrics_str + "\n", "a")

    def set_trial(self, trial):
        self.trial_log_path = os.path.join(os.path.dirname(self.log_path), f"{trial.number:0>3d}", f'{self.log_name}.txt')
        self.trial = trial

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
            self.save_to_txt(self.trial_log_path, metrics_str + "\n", "a")