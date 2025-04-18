import torch.nn as nn
import torch.nn.functional as F
import torch
from models.ban import *
from torch.nn.utils.weight_norm import weight_norm
from models.MLPDecoder import *
from models.ProteinCNN import *
from models.GraphDenseNET import *
def binary_cross_entropy(pred_output, label):
    class_output = F.log_softmax(pred_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1] 
    loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    return n, loss
def mean_squared_error(linear_output, label, weights=None):
    class_output = linear_output
    n = F.softmax(linear_output, dim=1)[:,0]
    loss = nn.MSELoss()
    loss = loss(class_output.view(-1), label.view(-1))
    return  n , loss
def cross_entropy_logits(logits, labels):
    loss = nn.CrossEntropyLoss()
    return loss(logits, labels)
def entropy_logits(logits):
    softmax = torch.nn.functional.softmax(logits, dim=1)
    log_softmax = torch.nn.functional.log_softmax(logits, dim=1)
    entropy = -torch.sum(softmax * log_softmax, dim=1)
    return entropy.mean()
class DrugBAN(nn.Module):
    def __init__(self, **config):
        super(DrugBAN, self).__init__()
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"] 
        num_filters = config["PROTEIN"]["NUM_FILTERS"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        out_binary = config["DECODER"]["BINARY"]
        ban_heads = config["BCN"]["HEADS"]
        filter_num=32
        self.drug_extractor = GraphDenseNet(num_input_features=22, growth_rate=32, out_dim=filter_num*4, 
                                            block_config=[8, 8, 8], bn_sizes=[2, 2, 2])
        self.protein_extractor = Predictor(Affine)
        self.bcn = weight_norm(
            BANLayer(v_dim=drug_hidden_feats[-1], q_dim=num_filters[-1], h_dim=mlp_in_dim, h_out=ban_heads),
            name='h_mat', dim=None)
        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)

    def forward(self, train_loader, mode="train"): 
        v_p = self.protein_extractor(train_loader)
        v_d = self.drug_extractor(train_loader)
        att = self.bcn(v_d, v_p,softmax = True)
        score = self.mlp_classifier(f)
        if mode == "train":
            return v_d, v_p, f, score
        elif mode == "eval":
            return v_d, v_p, score, att

class Affine(nn.Module):
    def __init__(self, dim):
        super(Affine, self).__init__()
        self.g = nn.Parameter(torch.ones(1, 1, dim))
        self.b = nn.Parameter(torch.zeros(1, 1, dim))
    def forward(self, x):
        return x * self.g + self.b
