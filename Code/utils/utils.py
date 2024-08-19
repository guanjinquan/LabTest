import matplotlib.pyplot as plt
import os
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score
import torch
import numpy as np
import random


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    

def get_metrics(y_true, y_prob):
    y_pred = [0 if i < 0.5 else 1 for i in y_prob]  # 二分类取最大值
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    auc_value = roc_auc_score(y_true, y_prob)
    return {"acc": round(accuracy, 4), "recall": round(recall, 4), "f1": round(f1, 4), "precision": round(precision, 4), "auc": round(auc_value, 4)}


def Draw(result_path, metrics): 
    # 绘制Loss
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15, 8))
    color = ['blue', 'orange', 'green', 'red']
    
    x1 = range(0, len(metrics['train']['loss']))
    x2 = range(0, len(metrics['test']['loss']))
    
    # 6个子图 + x轴
    axes = [ax1, ax2, ax3, ax4, ax5, ax6]
    xs = [x1, x2]
    
    # 绘制
    for col_id, mode in enumerate(metrics.keys()):
        if mode == 'train':
            x = xs[0]
        else:
            x = xs[1]
        for met in metrics[mode].keys():
            if met == 'loss':
                axes[0].plot(x, metrics[mode][met], label=f'{mode} {met}', color=color[col_id])
            elif met == 'acc':
                axes[1].plot(x, metrics[mode][met], label=f'{mode} {met}', color=color[col_id])
            elif met == 'f1':
                axes[2].plot(x, metrics[mode][met], label=f'{mode} {met}', color=color[col_id])
            elif met == 'precision':
                axes[3].plot(x, metrics[mode][met], label=f'{mode} {met}', color=color[col_id])
            elif met == 'recall':
                axes[4].plot(x, metrics[mode][met], label=f'{mode} {met}', color=color[col_id])
            elif met == 'auc':
                axes[5].plot(x, metrics[mode][met], label=f'{mode} {met}', color=color[col_id])
     
    for ax in axes:
        ax.legend()  # 显示图例
     
    plt.savefig(os.path.join(result_path, f'result.png'))
    plt.close(fig)