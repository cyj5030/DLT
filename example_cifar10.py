import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import torchvision

from datasets import load_dataset

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
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

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
    ''' <<<<<<<<<<< setup >>>>>>>>>>> '''
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

    ''' <<<<<<<<<<< load dataset >>>>>>>>>>> '''
    cifar10 = load_dataset('cifar10')
    train_set = CIFAR10(cifar10["train"])
    val_set = CIFAR10(cifar10["test"])
    
    ''' <<<<<<<<<<< dataloader >>>>>>>>>>> '''
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=0)

    ''' <<<<<<<<<<< network >>>>>>>>>>> '''
    network = Net()

    ''' <<<<<<<<<<< loss >>>>>>>>>>> '''
    loss_fn = Loss()

    ''' <<<<<<<<<<< optimizer and lr_scheduler >>>>>>>>>>> '''
    lr = 0.001
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    n_epochs = 10
    total_steps = n_epochs * len(train_loader)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=10*lr, total_steps=total_steps)
    
    ''' <<<<<<<<<<< seed >>>>>>>>>>> '''
    from accelerate.utils import set_seed
    set_seed(42)

    ''' <<<<<<<<<<< metrics >>>>>>>>>>> '''
    metrics = {"accuracy": accuracy}

    ''' <<<<<<<<<<< callback >>>>>>>>>>> '''
    from framework.common_callback import EpochCheckpoint, EpochTrainLog, EpochEvalLog
    log_folder = "./examples/train_log/cifar10"
    weights_folder = os.path.join(log_folder, "weights")
    os.makedirs(weights_folder, exist_ok=True)
    callbacks = [
        EpochCheckpoint(weights_folder, save_freq=1),
        EpochTrainLog(log_folder, save_freq=1),
        EpochEvalLog(log_folder, save_freq=1),
    ]
    
    from framework.model import Model
    model = Model(network, loss_fn, optimizer, metrics_dict=metrics, lr_scheduler=lr_scheduler)
    history = model.fit_ddp(train_loader, 
                    val_loader, 
                    epochs=n_epochs, 
                    ckpt_path=os.path.join(weights_folder, "checkpoint.pth"), 
                    callbacks=callbacks,
                    early_stopping=True,
                    patience=n_epochs - 5, 
                    monitor="val_accuracy",
                    mode="max"
                    )
                    