import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import torchvision

from datasets import load_dataset

import sys
sys.path.append(os.getcwd())

class CIFAR10(torch.utils.data.Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features, labels = {}, {}
        features["image"] = torchvision.transforms.ToTensor()(self.data[idx]['img'])
        labels["labels"] = self.data[idx]["label"]
        return features, labels

class Net(nn.Module):
    def __init__(self, hidden_planes=16, hidden_dims=128):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, hidden_planes, 5)
        self.fc1 = nn.Linear(hidden_planes * 5 * 5, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, hidden_dims)
        self.fc3 = nn.Linear(hidden_dims, 10)

    def forward(self, input_data):
        x = input_data["image"]
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, features, label):
        loss = self.ce(features, label["labels"])
        return loss

def accuracy(network_outputs, data_labels):
    logits = network_outputs
    labels = data_labels["labels"]
    accurate_preds = (logits.argmax(-1) == labels).float().mean().item()
    return accurate_preds

if __name__ == "__main__":
    import framework.hyperSearch as HPS

    ''' <<<<<<<<<<< setup >>>>>>>>>>> '''
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    ''' <<<<<<<<<<< load dataset >>>>>>>>>>> '''
    cifar10 = load_dataset('cifar10')
    train_dataset_fn = lambda : CIFAR10(cifar10["train"])
    val_dataset_fn = lambda : CIFAR10(cifar10["test"])
    hps_train_loader = HPS.DataSpace(train_dataset_fn, 
                              {}, 
                              {"batch_size": [16, 32, 64], "shuffle": [True], "num_workers": [2]})
    hps_val_loader = HPS.ValDataSpace(val_dataset_fn, 
                               {},
                               {"batch_size": [32], "shuffle": [False], "num_workers": [0]})

    ''' <<<<<<<<<<< network >>>>>>>>>>> '''
    hps_network = HPS.ModuleSpace(Net, {"hidden_planes": [16, 24, 32], "hidden_dims": [128]})

    ''' <<<<<<<<<<< loss >>>>>>>>>>> '''
    hps_loss = HPS.ModuleSpace(Loss, {})

    ''' <<<<<<<<<<< optimizer and lr_scheduler >>>>>>>>>>> '''
    hps_epoch = [10]
    import numpy as np
    hps_seed = np.random.randint(0, 100, (10)).tolist()

    hps_optim = HPS.OptimSpace(["AdamW"], 
                               {"lr": [1e-4, 2e-4]})
    hps_lr_scheduler = HPS.LrSchedulerSpace(["OneCycleLR"], 
                                            {"max_lr": [HPS.HpsMul(10, "lr")], "total_steps": [HPS.HpsAdd(0, "total_steps")]})
    
    ''' <<<<<<<<<<< metrics >>>>>>>>>>> '''
    hps_metrics = {"accuracy": accuracy}

    log_folder = "./examples/train_log"
    hps_task = HPS.HyperParamSearch(
        network = hps_network, 
        loss_fn = hps_loss,
        optimizer = hps_optim,
        train_dataset = hps_train_loader, 
        val_dataset = hps_val_loader,
        epoch_space = hps_epoch,
        lr_scheduler = hps_lr_scheduler,
        metrics_dict = hps_metrics,
        seed_space = hps_seed,
        log_folder = log_folder)

    study_name = "hps_cifar10"
    from framework.hps_callback import EpochEvalLog, EpochTrainLog, EpochCheckpoint, Pruning
    save_folder = os.path.join(log_folder, study_name)
    callbacks = [
        Pruning("val_accuracy"),
        EpochCheckpoint(os.path.join(save_folder, "weights"), save_freq=1),
        EpochTrainLog(save_folder, save_freq=1),
        EpochEvalLog(save_folder, save_freq=1),
    ]
    hps_task.fit(study_name, callbacks, monitor="val_accuracy", direction="maximize", n_trials=4, pruner=None)