from models.main_model import * 
from time import time 
from utils import set_seed, graph_collate_func, mkdir 
from configs import get_cfg_defaults 
from torch.utils.data import DataLoader 
from trainer import Trainer
import torch
import argparse
import warnings, os  
from dataset import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description="DrugBAN for DTI prediction")
parser.add_argument('--cfg', required=True, help="path to config file", type=str)
parser.add_argument('--data', required=True, type=str, metavar='TASK',
                    help='dataset')
args = parser.parse_args()

def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    set_seed(cfg.SOLVER.SEED)
    mkdir(cfg.RESULT.OUTPUT_DIR)
    experiment = None
    print(f"Config yaml: {args.cfg}") 
    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")
    dataFolder = f'./datasets/{args.data}'
    model = DrugBAN(**cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
    torch.backends.cudnn.benchmark = True
    trainer = Trainer(model, opt, device, fpath = dataFolder,experiment=experiment, **cfg)
    result = trainer.train()
    with open(os.path.join(cfg.RESULT.OUTPUT_DIR, "model_architecture.txt"), "w") as wf:
        wf.write(str(model))
    print(f"Directory for saving result: {cfg.RESULT.OUTPUT_DIR}")
    return result
if __name__ == '__main__':
    s = time()
    result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")
