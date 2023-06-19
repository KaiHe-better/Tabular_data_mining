import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--ID', default='0', help='run ID')
parser.add_argument('--gpu', default='3', type=str, help='gpu device numbers')
parser.add_argument('--load_ckpt', default=None, help='load saved ckpt')

parser.add_argument('--batch_size', type=int, default=16,help='batch_size')
parser.add_argument('--grad_iter', type=int, default=1,help='batch_size')
parser.add_argument('--lr', type=float, default=1e-5,help='learning rate')
parser.add_argument('--train_file_path', type=str, default="./datasets/raw_data/In-vehicleCouponRecommendation/in-vehicle-coupon-recommendation.csv", help='train_file_path')

parser.add_argument('--max_cell_len', type=int, default=15,help='max_cell_len, token level')
parser.add_argument('--shift', type=int, default=None,help='max shift num , start from 1, none mean all cloumns are shifted')
parser.add_argument('--shift_ratio', type=float, default=0.3,help='random seed')
parser.add_argument('--discriminator_ratio', type=float, default=0.2,help='random seed')

parser.add_argument('--task_list', type=list, default=["1", "2"], help='random seed')
parser.add_argument('--shift_loss_weight', type=float, default=1,help='random seed')
parser.add_argument('--discriminator_loss_weight', type=float, default=0.5,help='random seed')

parser.add_argument('--inter_num', type=int, default=50,help='how many iter for averaging results')
parser.add_argument('--epoch', type=int, default=100,help='random seed')
parser.add_argument('--save_ckpt', type=bool, default=True,help='saving ckpt')
parser.add_argument('--save_res', type=bool, default=True,help='save_res')

parser.add_argument('--seed', type=int, default=666,help='random seed')
args = parser.parse_args()




import os
import sys
import numpy as np
import json
import torch
import random
import copy
import torch
from transformers import RobertaConfig, RobertaModel, RobertaTokenizerFast
from models.my_model import My_Model    
from utils.trainer import My_Trainer    
from utils.data_loader import get_loader   
   
main_gpu = int(args.gpu.split(",")[0])
torch.cuda.set_device(main_gpu)  
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# if args.load_ckpt is not None:
#     args.load_ckpt = "./ckpt/saved_ckpt/"+args.load_ckpt
    

def main(args):
    print("init model...")
    sentence_encoder = RobertaModel.from_pretrained("roberta-large", cache_dir="./ckpt/roberta")
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large", cache_dir="./ckpt/roberta")

    print("loading data...")
    train_data_loader, args = get_loader(args, tokenizer)
    trainer = My_Trainer(args, train_data_loader=train_data_loader, tokenizer=tokenizer, device=device)

    my_model = My_Model(args, sentence_encoder=sentence_encoder, tokenizer=tokenizer, device=device)
    if torch.cuda.is_available():
        my_model.to(device) 
    
    trainer.train(my_model)

if __name__ == "__main__":
    main(args)
