from datasets import GetDataLoader
from models import GetModel
from settings import GetOptimizer, GetScheduler, GetLoss
from utils import Logger, save_trainer, save_model, get_metrics, parse_arguments, Draw
import os
import tqdm
import numpy as np
import torch


def get_score(metrics):
    score = 0.6 * metrics[f"auc"] + 0.1 * metrics[f"f1"] + 0.3 * metrics[f"acc"]
    return score

        
class Trainer:
    def __init__(self, args=None):  
        self.args = parse_arguments() if args is None else args  # args可能是None，得用self.args
        
        # dataset 
        self.train_loader, self.val_loader, \
            self.test_loader = GetDataLoader(self.args)
        
        # running setting
        self.loss_fn = GetLoss(self.args)
        self.model = GetModel(self.args).cuda()
        self.optimizer = GetOptimizer(self.args, self.model)
        self.scheduler  = GetScheduler(self.args, self.optimizer)
        self.epoch = 0
        self.best_metrics = {}
        self.best_score = 0
        self.acc_step = self.args.acc_step  # accumulate_step
        
        # trainer config
        run_path = [self.args.model, self.args.runs_id]
        self.log_path = os.path.join(self.args.log_path, *run_path)
        self.ckpt_path = os.path.join(self.args.ckpt_path, *run_path) 
        print("log_path : ", self.log_path, flush=True)
        print("ckpt_path : ", self.ckpt_path, flush=True)
        
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.ckpt_path, exist_ok=True)
        if os.path.exists(os.path.join(self.ckpt_path, 'Final_Trainer.pkl')):
            raise ValueError("Trainer.pkl exists!!! [Delete the checkpoint folder if you need to rerun]")  # 防止覆盖之前的训练记录
        
        self.log = Logger(os.path.join(self.log_path, 'log.txt'))
        self.log.write("settings : " + str(args))
        
        self.record_dict = {}
        self.record_dict['train'] = {'loss': [], 'auc': [], 'f1': [], 'acc': [], 'precision': [], 'recall': []}
        self.record_dict['valid'] = {'loss': [], 'auc': [], 'f1': [], 'acc': [], 'precision': [], 'recall': []}
        self.record_dict['test'] = {'loss': [], 'auc': [], 'f1': [], 'acc': [], 'precision': [], 'recall': []}

    
    def run(self):
        start_epoch = self.epoch
        
        for epoch_id in range(start_epoch, self.args.num_epochs + 1):  
            if epoch_id > start_epoch:  # 第0个epoch验证未训练的模型的性能
                self.train_epoch(self.train_loader)
                self.scheduler.step()
            else:
                self.eval_epoch(self.train_loader, 'train')
            self.eval_epoch(self.val_loader, 'valid')
            self.eval_epoch(self.test_loader, 'test')
            self.on_epoch_end()
    
    
    def train_epoch(self, train_loader):
        self.model.train()
        with tqdm.tqdm(total=len(train_loader)) as pbar:
            loss_list = []
            prob_list = []
            true_list = []
            self.optimizer.zero_grad()            
            for i, (x, y) in enumerate(train_loader, 1):
                x, y = x.cuda(), y.cuda()
                
                out = self.model(x)
                loss = self.loss_fn(out, y)
                probs = torch.softmax(out, dim=1)[:, 1].detach().cpu().numpy().tolist()
                
                loss_list.append(loss.cpu().detach().item())
                prob_list.extend(probs)
                true_list.extend(y.cpu().numpy().tolist())
                
                loss /= self.acc_step
                loss.backward()
                if i % self.acc_step == 0 or i == len(train_loader):  # i starts from 1
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                pbar.update(1)
                
            
            mean_loss = np.mean(loss_list)
            metrics_dict = get_metrics(true_list, prob_list)
            self.log.write(f"train_with_sampler epoch_{self.epoch} : {mean_loss}")
            self.log.write(f'metrics : ' + str(metrics_dict))
            self.record_dict['train']['loss'].append(mean_loss)
            for k, v in metrics_dict.items():
                self.record_dict['train'][k].append(v)
            pbar.close()
        
        metrics_dict['loss'] = mean_loss
        return metrics_dict
    
    
    def eval_epoch(self, val_loader, mode='valid'):
        self.model.eval()
        with torch.no_grad():
            with tqdm.tqdm(total=len(val_loader)) as pbar:
                loss_list = []
                prob_list = []
                true_list = []    
                       
                for i, (x, y) in enumerate(val_loader, 1):
                    x, y = x.cuda(), y.cuda()
                    out = self.model(x)
                    loss = self.loss_fn(out, y)
                    probs = torch.softmax(out, dim=1)[:, 1].detach().cpu().numpy().tolist()
                    loss_list.append(loss.cpu().detach().item())
                    prob_list.extend(probs)
                    true_list.extend(y.cpu().numpy().tolist())
                    pbar.update(1)

                mean_loss = float(np.mean(loss_list))
                metrics_dict = get_metrics(true_list, prob_list)
                self.log.write(f"{mode} epoch_{self.epoch} : {mean_loss}")
                self.log.write(f'metrics : ' + str(metrics_dict))
                self.record_dict[mode]['loss'].append(mean_loss)
                for k, v in metrics_dict.items():
                    self.record_dict[mode][k].append(v)
            pbar.close()
            
        # save best model of valid
        if mode in ['valid']:
            score = get_score(metrics_dict)
            if score > self.best_score:
                self.best_score = score
                self.best_metrics = metrics_dict
                save_model(self.model, self.epoch, os.path.join(self.ckpt_path, f'{mode}_Best.pth'))
                
    
    def on_epoch_end(self):
        # draw loss and metrics
        Draw(self.log_path, self.record_dict)
        
        # save model
        save_trainer(self, os.path.join(self.ckpt_path, 'Final_Trainer.pkl'))
        save_model(self.model, self.epoch, os.path.join(self.ckpt_path, f'Final.pth'))
        self.epoch += 1  
    