import pickle
import timeit
import numpy as np
from math import sqrt
import math
import torch
import torch.optim as optim
import os
from torch import nn, einsum  
from torch.nn import Parameter
import torch.nn.functional  as F
from sklearn.metrics import mean_squared_error
from torch.nn import init
import os
torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True
device = "cuda" 
class atten(nn.Module):
    def __init__(self, padding=3):
        super(atten, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(2 * padding + 1), padding=padding,
                               bias=False)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        xc = x.unsqueeze(1)
        xt = torch.cat((xc, self.dropout(xc)), dim=1)
        att = self.sigmoid(self.conv1(xt))
        return att.squeeze(1)

class spatt(nn.Module):
    def __init__(self, padding = 3):
        super(spatt, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2,out_channels=1,kernel_size=(2*padding+1),padding=padding,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xc = x.unsqueeze(1)
        avg = torch.mean(xc, dim=1, keepdim=True)
        max_x, _ = torch.max(xc, dim=1, keepdim=True)
        xt = torch.cat((avg,max_x),dim=1)
        att = self.sigmoid(self.conv1(xt))
        return x * (att.squeeze(1))

class Affine(nn.Module):
    def __init__(self, dim):
        super(Affine, self).__init__()
        self.g = nn.Parameter(torch.ones(1, 1, dim))
        self.b = nn.Parameter(torch.zeros(1, 1, dim))
    def forward(self, x):
        return x * self.g + self.b

class self_attention(nn.Module):
    def __init__(self, channel):
        super(self_attention, self).__init__()
        self.linear_Q = nn.Linear(channel, channel)
        self.linear_K = nn.Linear(channel, channel)
        self.linear_V = nn.Linear(channel, channel)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, xs):
        Q = self.linear_Q(xs)
        K = self.linear_K(xs)
        scale = K.size(-1) ** -0.5
        att = self.softmax(Q * scale)
        ys = att * K
        return ys

class spatial_attention(nn.Module):
    def __init__(self, padding=3):
        super(spatial_attention, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=(2 * padding + 1), padding=padding,
                               bias=False)
        self.dropout = nn.Dropout(0.1)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_x, _ = torch.max(x, dim=1, keepdim=True)
        xt = torch.cat((avg, max_x), dim=1)
        att = self.Sigmoid((self.conv1(xt)))
        return att.expand_as(x)

class CNN_MLP(nn.Module):
    def __init__(self, Affine, patch, channel, output_size, dr, down=False, last=False):
        super(CNN_MLP, self).__init__()
        self.Afine_p1 = Affine(channel)
        self.Afine_p2 = Affine(channel)
        self.Afine_p3 = Affine(channel)
        self.Afine_p4 = Affine(channel)
        self.cross_patch_linear0 = nn.Linear(patch, patch)
        self.cross_patch_linear1 = nn.Linear(patch, patch)
        self.cross_patch_linear = nn.Linear(patch, patch)
        self.attention_patch_linear2 = nn.Linear(patch, patch)
        self.cnn1 = nn.Conv1d(in_channels=patch, out_channels=patch, kernel_size=15, padding=7, groups=patch)
        self.bn1 = nn.BatchNorm1d(patch)
        self.cnn2 = nn.Conv1d(in_channels=patch, out_channels=patch, kernel_size=31, padding=15, groups=patch)
        self.bn2 = nn.BatchNorm1d(patch)
        self.cnn3 = nn.Conv1d(in_channels=patch, out_channels=patch, kernel_size=7, padding=3, groups=patch)
        self.bn3 = nn.BatchNorm1d(patch)
        self.attention_patch_linear2 = nn.Linear(patch, patch)
        self.self_attention = self_attention(channel)
        self.bnp1 = nn.BatchNorm1d(channel)
        self.cross_channel_linear1 = nn.Linear(channel, channel)
        self.cross_channel_linear2 = nn.Linear(channel, channel)
        self.att = spatt(3)
        self.att_sp = spatial_attention(3)
        self.attention_channel_linear2 = nn.Linear(channel, channel)
        self.last_linear = nn.Linear(channel, output_size)
        self.bnp = nn.BatchNorm1d(patch)
        self.act = nn.ReLU()
        self.last = last
        self.dropout = nn.Dropout(0.05)
        self.down = down

    def forward(self, x):
        x_cp = self.Afine_p1(x).permute(0, 2, 1)
        x_cp = self.act(self.cross_patch_linear0(x_cp))
        x_cp = self.act(self.cross_patch_linear1(x_cp))
        x_cp = self.cross_patch_linear(x_cp).permute(0, 2, 1)
        x_cc = x + self.Afine_p2(x_cp)
        x_cc2 = self.Afine_p3(x_cc)
        x_cc2 = self.act(self.bn1(self.cnn1(x_cc2)))
        x_cc2 = self.act(self.bn2(self.cnn2(x_cc2)))
        x_cc2 = self.act(self.bn3(self.cnn3(x_cc2)))
        x_cc2 = self.Afine_p4(x_cc2)
        x_cc2 = self.att(x_cc2)
        x_out = x_cc + self.dropout(x_cc2)
        if self.last == True:
            x_out = self.last_linear(x_out)
        return x_out

class sequence_embedding(nn.Module):
    def __init__(self,):
        super(sequence_embedding, self).__init__()
        self.pool = nn.MaxPool1d(12)
        self.softmax = nn.Softmax(-1)

class Predictor(nn.Module):
    def __init__(self,  Affine):
        super(Predictor, self).__init__()
        self.embed_word = nn.Embedding(264, 100)
        self.embed_ss = nn.Embedding(512, 100)
        self.sequence_embedding = sequence_embedding()
        self.WP_NN1 = CNN_MLP(Affine, 1200, 100, 128, 0, True)
        self.WP_NN3 = CNN_MLP(Affine, 1200, 100, 128, 0, True)
        self.WP_NN2 = CNN_MLP(Affine, 1200, 100, 128, 0, True, True)
        self.WP_FiNN1 = CNN_MLP(Affine, 1200, 100, 128, 0,True)
        self.WP_FiNN3 = CNN_MLP(Affine, 1200, 100, 128, 0, True)
        self.WP_FiNN2 = CNN_MLP(Affine, 1200, 100, 128, 0,True, True)
        self.WF = nn.Linear(128, 128)
        self.WS = nn.Linear(128, 128)
        self.merge_atten2 = atten(3)
        self.down_sample2 = nn.Linear(256, 128)
        self.activation = nn.ReLU()

    def Elem_feature_Fusion_P(self, xs, x):
        x_c = self.down_sample2(torch.cat((xs, x), dim=2))
        x_c = self.merge_atten2(x_c)
        xs_ = self.activation(self.WF(xs))
        x_ = self.activation(self.WS(x))
        xs_m = x_c * xs_ + xs
        ones = torch.ones(x_c.shape).to(device)
        x_m = (ones - x_c) * x_ + x
        ys = xs_m + x_m
        return ys
    
    def forward(self, inputs):
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 
        words, protein_ss = inputs.protein_feature,inputs.protein_AnJiSuan    
        N = words.shape[0]
        word_vectors = torch.zeros((words.shape[0], words.shape[1], 100)).to("cuda")
        for i in range(N):                   
            t = self.embed_word(torch.LongTensor(words[i].to('cpu').numpy()).cuda())
            tf = F.normalize(t, dim=1)
            word_vectors[i, :, :] = tf
        protein_fi_vector = torch.zeros((protein_ss.shape[0], protein_ss.shape[1], 100)).to("cuda")
        for i in range(N):           
            t = self.embed_ss(torch.LongTensor(protein_ss[i].to('cpu').numpy()).cuda())           
            tf = F.normalize(t, dim=1)  
            protein_fi_vector[i, :, :] = tf
        protein_vector = word_vectors + self.WP_NN1(word_vectors)
        protein_vector = protein_vector + self.WP_NN3(protein_vector)
        protein_vector = self.WP_NN2(protein_vector)
        protein_fi_vector = protein_fi_vector + self.WP_FiNN1(protein_fi_vector)
        protein_fi_vector = protein_fi_vector + self.WP_FiNN3(protein_fi_vector)
        protein_fi_vector = self.WP_FiNN2(protein_fi_vector)
        protein_vectors = self.Elem_feature_Fusion_P(protein_vector, protein_fi_vector)
        return protein_vectors