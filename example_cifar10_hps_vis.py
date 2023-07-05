import optuna

import sys, os
sys.path.append(os.getcwd())

from framework.utils import pkl_save, pkl_load

if __name__ == "__main__":
    study = pkl_load("/home/cyj/code/dl_framework/examples/train_log/hps_cifar10/hps_cifar10.pkl")

    # 各种图例可以参考 https://tigeraus.gitee.io/doc-optuna-chinese-build/reference/visualization.html
    fig = optuna.visualization.plot_parallel_coordinate(study, params=['dataset.batch_size', 'module.hidden_planes'])