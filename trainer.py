'''
这个里面不完全独立，目前用的是回归方案里面的，需要修改的只需要修改下面相同的被标注出来的地方。
'''

import torch
import torch.nn as nn
import copy
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix
from sklearn.metrics import precision_recall_curve, precision_score
from models.main_model import binary_cross_entropy, cross_entropy_logits, entropy_logits
from prettytable import PrettyTable
from domain_adaptator import ReverseLayerF
from tqdm import tqdm
from dataset import *
from preprocessing import GNNDataset
from torch.utils.data import DataLoader #torch自带的数据加载器//这个地方可能不能用
from math import sqrt
from torch_geometric.data import DataLoader
from scipy import stats

class Trainer(object):
    def __init__(self, model, optim, device,  fpath=None , discriminator=None,
                 experiment=None, alpha=1, **config):
        self.fpath = fpath  
        train_set = GNNDataset(fpath, Type="train")
        val_set = GNNDataset(fpath, Type="val")  
        test_set = GNNDataset(fpath, Type="test")  
        print(len(train_set))
        print(len(test_set))
        print(len(val_set))
        train_loader =DataLoader(train_set, batch_size=32, shuffle=True, num_workers=8)
        test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=8)
        val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=8)
        self.model = model
        self.optim = optim
        self.device = device
        self.epochs = config["SOLVER"]["MAX_EPOCH"]
        self.current_epoch = 0 
        self.train_dataloader = train_loader
        self.val_dataloader = val_loader
        self.test_dataloader = test_loader
        self.is_da = config["DA"]["USE"]
        self.alpha = alpha
        self.n_class = config["DECODER"]["BINARY"]    
        self.batch_size = config["SOLVER"]["BATCH_SIZE"]
        self.nb_training = len(self.train_dataloader)
        self.step = 0 
        self.experiment = experiment
        self.best_model = None 
        self.best_mse = 1000
        self.best_ci = 0
        self.best_model = None
        self.best_epoch = None
        self.best_auroc = 0
        self.val_auroc_epoch = []
        self.train_loss_epoch = []
        self.train_model_loss_epoch = []
        self.val_loss_epoch= []
        self.test_metrics = {}
        self.config = config 
        self.output_dir = config["RESULT"]["OUTPUT_DIR"]
        valid_metric_header = ["# Epoch", "rmse", "mse", "pearson","spearman","ci","cm","Val_loss"]#验证集
        test_metric_header = ["# Best Epoch", "rmse", "mse", "pearson","spearman","ci","cm","test_loss"]#测试集
        train_metric_header = ["# Epoch", "Train_loss"]
        '''
        valid_metric_header = ["# Epoch", "AUROC", "AUPRC", "Val_loss"]
        test_metric_header = ["# Best Epoch", "AUROC", "AUPRC", "F1", "Sensitivity", "Specificity", "Accuracy",
                              "Threshold", "Test_loss"]
        train_metric_header = ["# Epoch", "Train_loss"]'''
        self.val_table = PrettyTable(valid_metric_header)
        self.test_table = PrettyTable(test_metric_header)
        self.train_table = PrettyTable(train_metric_header)

    def train(self):
        float2str = lambda x: '%0.4f' % x
        for i in range(self.epochs):
            self.current_epoch += 1 
            train_loss,score,labels = self.train_epoch()
            train_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [train_loss]))
            self.train_table.add_row(train_lst)
            self.train_loss_epoch.append(train_loss)
            G,P,val_loss = self.test(dataloader="val")
            rmse,mse,pearson,spearman,ci,cm = self.rmse(G,P),self.mse(G,P),self.pearson(G,P),
            self.spearman(G,P),self.ci(G,P),get_rm2(G,P)         
            val_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [rmse ,mse,pearson,
                                                                                  spearman,ci,cm,val_loss]))
            self.val_table.add_row(val_lst)
            self.val_loss_epoch.append(val_loss)  
            if mse < self.best_mse:
                self.best_model = copy.deepcopy(self.model)
                self.best_mse = mse
                self.best_ci = ci           
                self.best_epoch = self.current_epoch
            print('Validation at Epoch ' + str(self.current_epoch) + ' with validation loss ' + str(val_loss), "mse"
                  + str(mse) + " ci " + str(ci))
        G,P,test_loss = self.test(dataloader="test")
        rmse,mse,pearson,spearman,ci,cm = self.rmse(G,P),self.mse(G,P),self.pearson(G,P),
        self.spearman(G,P),self.ci(G,P),get_rm2(G,P)
        test_lst = ["epoch " + str(self.best_epoch)] + list(map(float2str, [rmse,mse,pearson,
                                                                            spearman,ci,cm,test_loss]))
        self.test_table.add_row(test_lst)
        print('Test at Best Model of Epoch ' + str(self.best_epoch) + ' with test loss ' + str(test_loss), " rmse "
              + str(rmse) + " mse " + str(mse) + " ci " + str(ci) + " pearson " +
              str(pearson)+ " cm2 " +str(cm) )
        self.test_metrics["rmse"] = rmse
        self.test_metrics["mse"] = mse
        self.test_metrics["pearson"] = pearson
        self.test_metrics["spearman"] = spearman
        self.test_metrics["ci"] = ci
        self.test_metrics["cm"] = cm
        self.test_metrics["test_loss"] = test_loss
        self.test_metrics["best_epoch"] = self.best_epoch
        self.save_result()
        return self.test_metrics
    def save_result(self):
        if self.config["RESULT"]["SAVE_MODEL"]:
            torch.save(self.best_model.state_dict(),
                       os.path.join(self.output_dir, f"best_model_epoch_{self.best_epoch}.pth"))
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"model_epoch_{self.current_epoch}.pth"))
        state = {
            "train_epoch_loss": self.train_loss_epoch,
            "val_epoch_loss": self.val_loss_epoch,
            "test_metrics": self.test_metrics,
            "config": self.config
        }
        torch.save(state, os.path.join(self.output_dir, f"result_metrics.pt"))
        val_prettytable_file = os.path.join(self.output_dir, "valid_markdowntable.txt")
        test_prettytable_file = os.path.join(self.output_dir, "test_markdowntable.txt")
        train_prettytable_file = os.path.join(self.output_dir, "train_markdowntable.txt")
        with open(val_prettytable_file, 'w') as fp:
            fp.write(self.val_table.get_string())
        with open(test_prettytable_file, 'w') as fp:
            fp.write(self.test_table.get_string())
        with open(train_prettytable_file, "w") as fp:
            fp.write(self.train_table.get_string())

    def _compute_entropy_weights(self, logits):
        entropy = entropy_logits(logits)
        entropy = ReverseLayerF.apply(entropy, self.alpha)
        entropy_w = 1.0 + torch.exp(-entropy)
        return entropy_w

    def train_epoch(self):
        self.model.train()
        loss_epoch = 0
        num_batches = len(self.train_dataloader)
        i=0
        bitch_len = len(self.train_dataloader)
        for train_loader in self.train_dataloader:           
            i+=1
            if i == bitch_len:
                continue
            train_loader=train_loader.to(self.device)
            self.optim.zero_grad()
            v_d, v_p, f, score = self.model(train_loader)
            labels = train_loader.y.to("cuda")
            if self.n_class == 1:
                n , loss = binary_cross_entropy(score, labels)
            else:
                n , loss = cross_entropy_logits(score, labels)
            loss.backward()
            self.optim.step()
            loss_epoch += loss.item()
            if self.experiment:
                self.experiment.log_metric("train_step model loss", loss.item(), step=self.step)
        loss_epoch = loss_epoch / num_batches
        print('Training at Epoch ' + str(self.current_epoch) + ' with training loss ' + str(loss_epoch))
        return loss_epoch,score,labels

    def rmse(self,y,f):
        rmse = sqrt(((y - f)**2).mean(axis=0))
        return rmse
    def mse(self,y,f):
        mse = ((y - f)**2).mean(axis=0)
        return mse
    def pearson(self,y,f):
        rp = np.corrcoef(y, f)[0,1]
        return rp
    def spearman(self,y,f):
        rs = stats.spearmanr(y, f)[0]
        return rs
    def ci(self,y,f):
        ind = np.argsort(y)
        y = y[ind]
        f = f[ind]
        i = len(y)-1
        j = i-1
        z = 0.0
        S = 0.0
        while i > 0:
            while j >= 0:
                if y[i] > y[j]:
                    z = z+1
                    u = f[i] - f[j]
                    if u > 0:
                        S = S + 1
                    elif u == 0:
                        S = S + 0.5
                j = j - 1
            i = i - 1
            j = i-1
        ci = S/z
        return ci

    def test(self, dataloader="test"):
        test_loss = 0
        total_preds = torch.Tensor().cpu()
        total_labels = torch.Tensor().cpu()
        if dataloader == "test":
            data_loader = self.test_dataloader
        elif dataloader == "val":
            data_loader = self.val_dataloader
        else:
            raise ValueError(f"Error key value {dataloader}")
        
        num_batches = len(data_loader)
        with torch.no_grad():
            
            self.model.eval()
            i = 0
            for test_loader in data_loader:
                i+=1
                if i == num_batches:
                    continue
                test_loader=test_loader.to(self.device)
                labels = test_loader.y.to("cuda")
                if dataloader == "val":
                    v_d, v_p, f, score = self.model(test_loader)
                elif dataloader == "test":
                    v_d, v_p, f, score = self.best_model(test_loader)
                if self.n_class == 1:
                    mean_squared_error
                    n , loss = mean_squared_error(score, labels)
                else:
                    n , loss = binary_cross_entropy(score, labels)
                test_loss += loss.item()
                
                total_labels = torch.cat((total_labels,labels.cpu()), 0)
                total_preds = torch.cat((total_preds, score.cpu()), 0)
            test_loss = test_loss / num_batches
        return total_labels.numpy().flatten(),total_preds.numpy().flatten(),test_loss
   
def get_rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)
    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))    
    
def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]
    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult
    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))
    return mult / (float(y_obs_sq * y_pred_sq) + 0.00000001)

def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    return 1 - (upp / (float(down) + 0.00000001))

def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    return sum(y_obs * y_pred) / (float(sum(y_pred * y_pred)) + 0.00000001)
