import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--ID', default='0', help='run ID')
parser.add_argument('--gpu', default='2', type=str, help='gpu device numbers')
parser.add_argument('--load_ckpt', default=None, help='load saved ckpt')

parser.add_argument('--batch_size', type=int, default=4,help='batch_size')
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
import torch.nn as nn
import random
import copy
import torch
from transformers import RobertaConfig, RobertaModel, RobertaTokenizerFast
from models.my_model import My_Model    
from utils.trainer import My_Trainer    
from utils.data_loader import get_loader, get_loader2
from models.logreg import LogReg
   
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

if args.load_ckpt is not None:
    args.load_ckpt = "./ckpt/saved_ckpt/"+args.load_ckpt
    # args.load_ckpt = './results/output/[]/'+args.load_ckpt
    

def find_epoch(hid_units, nb_classes, train_embs, train_lbls, test_embs, test_lbls):
    
    log = LogReg(hid_units, nb_classes)
    opt = torch.optim.Adam(log.parameters(), lr=0.001, weight_decay=0) #0.00001
    xent = nn.CrossEntropyLoss()
    log.to(device) 

    epoch_flag = 0
    epoch_win = 0
    best_acc = torch.zeros(1).to(device) 

    for e in range(20000):
        log.train()
        opt.zero_grad()

        logits = log(train_embs)
        loss = xent(logits, train_lbls)
        
        loss.backward()
        opt.step()

        if (e+1)%50 == 0:
            log.eval()
            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1)
            acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            print ('find acc', acc)
            if acc >= best_acc:
                epoch_flag = e+1
                best_acc = acc
                epoch_win = 0
            else:
                epoch_win += 1
            if epoch_win == 10:
                break
    return epoch_flag


def main(args):

    data_split = np.load('./datasets/raw_data/In-vehicleCouponRecommendation/data_split.npy', allow_pickle=True).item()
    train_idx = data_split['train']
    test_idx = data_split['test']


    print("init model...")
    sentence_encoder = RobertaModel.from_pretrained("roberta-large", cache_dir="./ckpt/roberta")
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large", cache_dir="./ckpt/roberta")

    ret_data_loader_train, args, col_name = get_loader2(args, tokenizer, train_idx)
    ret_data_loader_test, args, col_name = get_loader2(args, tokenizer, test_idx)
    
    col_name = col_name[:-1]  # label must be in the last, other data set need modification
    trainer = My_Trainer(args, train_data_loader=ret_data_loader_train, col_name=col_name, tokenizer=tokenizer, device=device)
    
    my_model = My_Model(args, sentence_encoder=sentence_encoder, tokenizer=tokenizer, device=device)
    if torch.cuda.is_available():
        my_model.to(device) 
    
    data_logreg_train, data_logreg_test = trainer.retrieval(my_model, ret_data_loader_train, ret_data_loader_test)


    print ('done')

    #data_logreg_train['sent'] = data_logreg_train['sent'].reshape(-1,data_logreg_train['sent'].shape[1]*data_logreg_train['sent'].shape[2])
    #data_logreg_test['sent'] = data_logreg_test['sent'].reshape(-1,data_logreg_test['sent'].shape[1]*data_logreg_test['sent'].shape[2])


    # data_logreg_train['sent'] = torch.mean(data_logreg_train['sent'], dim=1)
    # data_logreg_test['sent'] = torch.mean(data_logreg_test['sent'], dim=1)

    xent = nn.CrossEntropyLoss()
    hid_units = data_logreg_train['sent'].shape[1]
    print (data_logreg_train['sent'].shape)
    nb_classes = 2

    train_embs = data_logreg_train['sent'].to(device) 
    train_lbls = data_logreg_train['y_label'].to(device) 
    test_embs = data_logreg_test['sent'].to(device) 
    test_lbls = data_logreg_test['y_label'].to(device) 

    accs = []

    iter_num = find_epoch(hid_units, nb_classes, train_embs, train_lbls, test_embs, test_lbls)
    print ('Best iter_num:', iter_num)

    for _ in range(50):
        log = LogReg(hid_units, nb_classes)
        # opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
        opt = torch.optim.Adam(log.parameters(), lr=0.001, weight_decay=0) #0.00001
        log.to(device) 

        total_loss = 0
        # best_acc = torch.zeros(1).to(device) 

        for _ in range(iter_num):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)
            
            loss.backward()
            opt.step()

            total_loss+=loss

        print("total_loss", total_loss/iter_num)

        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs.append(acc * 100)
        print('Acc:', acc * 100)

    accs = torch.stack(accs)
    print('Average accuracy:', accs.mean())
    print('STD:', accs.std())


if __name__ == "__main__":
    main(args)
