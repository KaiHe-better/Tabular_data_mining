import os
import sys
import time
import copy
import json
import torch
import shutil
import numpy as np
from tqdm import tqdm
import utils.data_loader as data_loader
from utils.metrics import My_Metrics
from torch.optim import AdamW

class My_Trainer:

    def __init__(self, args, train_data_loader, tokenizer, device):
        self.args = args
        self.device = device
        self.train_data_loader = train_data_loader
        self.tokenizer = tokenizer
        sys_path = str(sys.argv[1:])[1:-1].replace("'", "").replace("--", "").replace(",", "_").replace(" ", "") if len(str(sys.argv[1:]))>2 else str(sys.argv[1:])
        self.dir_path ="./results/output/"+sys_path[:80].replace("/","_")
        self.res_path = self.dir_path+"/results.txt"
        self.record_train_path = self.dir_path+"/records.txt"
        self.my_metrics = My_Metrics()
        
        if os.path.exists(self.dir_path):
            shutil.rmtree(self.dir_path)
        os.mkdir(self.dir_path)

        with open(os.path.join(self.dir_path, "config.json"), "w") as f:
            tmp_dic = vars(args)
            json.dump(tmp_dic, f, indent=2)
                
    def init_model(self, my_model):
        parameters_to_optimize = list(my_model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        parameters_to_optimize = [
            {'params': [p for n, p in parameters_to_optimize
                        if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in parameters_to_optimize
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        
        optimizer = AdamW(parameters_to_optimize, lr=self.args.lr)

        # load model
        if self.args.load_ckpt is not None:
            print("loading ckpt...")
            if os.path.isfile(self.args.load_ckpt):
                state_dict = torch.load(self.args.load_ckpt, map_location=self.device)["state_dict"]
                own_state = my_model.state_dict()
                # print("own_state.keys()", list(own_state.keys()))
                for name, param in state_dict.items():
                    #name = name.replace("module.", "")
                    if name not in own_state.keys():
                        print('ignore {}'.format(name))
                    else:
                        #print('load {} from {}'.format(name, self.args.load_ckpt))
                        own_state[name].copy_(param)
                print("Successfully loaded checkpoint '%s'" % self.args.load_ckpt)
            else:
                raise Exception("No checkpoint found at '%s'" % self.args.load_ckpt)
        else:
            print("training from scratch...")
            
        return my_model, optimizer

    def recored_train_fn(self, record_train_path,  epoch_iter, it, best_iter, loss, shift_task_loss, discriminator_task_loss, 
                        shift_task_acc, discriminator_task_acc, discriminator_task_mse, averaged_acc, best_averaged_acc):
        with open(record_train_path, "a") as f:
            f.write("epoch_iter: {:4d}, it: {:7d}, loss {:5.6f}, shift_task_loss: {:5.6f}, discriminator_task_loss: {:5.6f}, shift_task_acc: {:5.6f}, discriminator_task_acc: {:5.6f}, discriminator_task_mse: {:5.6f}, best_averaged_acc: {:2.2f} (it:{:4d}) \n".format(\
                     epoch_iter,  it,  round(float(loss),6), round(float(shift_task_loss),6), round(float(discriminator_task_loss),6), 
                     round(shift_task_acc,6), round(discriminator_task_acc,6), round(discriminator_task_mse,6), round(best_averaged_acc, 6), best_iter))
            
    def recored_res_fn(self, shift_task_pred, shift_task_label, discriminator_task_pos_pred, discriminator_task_neg_pred):
         with open(self.res_path, "w") as f:
            for pred_item, label_item in zip(shift_task_pred, shift_task_label):
                f.write("pred_item  : "+str( ["{:2d}".format(i) for i in  pred_item.tolist() ] )+"\n")
                f.write("label_item : "+str( ["{:2d}".format(i) for i in  label_item.tolist()] )+"\n")
                f.write("discriminator_task_pos_pred (1): "+str( ["{:2f}".format(i) for i in  torch.cat(discriminator_task_pos_pred).view(-1)[:50]] )+"\n")
                f.write("discriminator_task_neg_pred (0): "+str( ["{:2f}".format(i) for i in  torch.cat(discriminator_task_neg_pred).view(-1)[:50]] )+"\n")
                f.write("\n")
    
    def train(self, my_model):

        self.my_model, self.optimizer = self.init_model(my_model)
        self.my_model.train()

        # Training
        averaged_acc = 0
        best_averaged_acc = 0
        best_iter = 0
        it = 0
        shift_task_acc = 0
        discriminator_task_acc = 0
        discriminator_task_mse = 0

        print_shift_loss_list = []
        print_discr_loss_list = []
        shift_task_pred = []
        shift_task_label = []
        discriminator_task_pos_pred = []
        discriminator_task_neg_pred = []
        print("Start training...")
        
        for epoch_iter in range(self.args.epoch):
            print("epoch_iter", epoch_iter)
            
            tbar = tqdm(self.train_data_loader, total=len(self.train_data_loader), disable=False, desc="Training", ncols=170)
            for data_item in tbar:
                if torch.cuda.is_available():
                    data_item["sent"] = data_item["sent"].to(self.device)
                    data_item["shift_label"] = data_item["shift_label"].to(self.device)
                    data_item["discriminator_input_index"] = data_item["discriminator_input_index"].to(self.device)
                    data_item["input_len_list"] = data_item["input_len_list"].to(self.device)
                
                loss, shift_task_res, shift_task_loss, discriminator_task_res, discriminator_task_loss= self.my_model(data_item)
                loss.backward()
                
                if (it % self.args.grad_iter == 0):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                shift_task_pred.append(shift_task_res)
                shift_task_label.append(data_item["shift_label"])
                discriminator_task_pos_pred.append(discriminator_task_res[0].view(-1))
                discriminator_task_neg_pred.append(discriminator_task_res[1].view(-1))
                print_shift_loss_list.append(shift_task_loss)
                print_discr_loss_list.append(discriminator_task_loss)

                tbar.set_postfix_str("shift_loss: {:2.4f},  discr_loss: {:2.4f}, shift_acc: {:2.2f}, discr_acc: {:2.2f}, discr_mse: {:2.4f}, averaged_acc: {:2.2f}".\
                    format(np.mean(print_shift_loss_list), np.mean(print_discr_loss_list), shift_task_acc, discriminator_task_acc, discriminator_task_mse, averaged_acc) )
                it +=1
                
                if it%self.args.inter_num==0 and it >1:
                    shift_task_acc = self.my_metrics.metrics_shift_task_res(shift_task_pred, shift_task_label)
                    discriminator_task_acc, discriminator_task_mse = self.my_metrics.metrics_discriminator_task_res(discriminator_task_pos_pred, discriminator_task_neg_pred)
                    averaged_acc = (shift_task_acc+discriminator_task_acc)/2
                    
                    self.recored_train_fn(self.record_train_path, epoch_iter, it, best_iter, loss, shift_task_loss.data, discriminator_task_loss, 
                        shift_task_acc, discriminator_task_acc, discriminator_task_mse, averaged_acc, best_averaged_acc)

                    if best_averaged_acc <= averaged_acc:
                        best_iter = it
                        best_averaged_acc = averaged_acc

                        if self.args.save_res:
                           self.recored_res_fn(shift_task_pred, shift_task_label, discriminator_task_pos_pred, discriminator_task_neg_pred)
                        
                        if self.args.save_ckpt:
                            save_ckpt_path = "{}/{}.pt".format(self.dir_path, it)
                            torch.save({'state_dict': self.my_model.state_dict()}, save_ckpt_path) 
                    
                    shift_task_pred = []
                    shift_task_label = []
                    discriminator_task_pos_pred = []
                    discriminator_task_neg_pred = []
                    print_loss_list = []
                    torch.cuda.empty_cache()
                    
            print("\n\n")
      
