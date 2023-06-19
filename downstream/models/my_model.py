
import torch
import itertools
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class My_Model(nn.Module):
    def __init__(self, args, sentence_encoder, tokenizer, device):
        nn.Module.__init__(self)
        self.args = args
        self.sentence_encoder = sentence_encoder
        self.tokenizer = tokenizer
        self.device = device
        self.dropout = nn.Dropout(p=0.5)
        
        self.num_class = 2
        self.shift_task_linear_layer = nn.Linear(1024, args.shift+1 if args.shift is not None else args.num_columns)
        self.discriminator_task_linear_layer = nn.Bilinear(1024, 1024, 1)
        
        self.shift_task_loss = nn.CrossEntropyLoss()
        self.discriminator_task_loss = nn.BCELoss()
    
    def forward(self, data_item):
        input_sent = data_item["sent"]
        bz, col_num, cell_len = input_sent.size()
        attention_mask = (input_sent != self.tokenizer.pad_token_id).bool().reshape(bz,-1).to(self.device) 
        sent_embed = self.sentence_encoder(input_sent.reshape(bz,-1), attention_mask=attention_mask).last_hidden_state
        # sent_embed = self.dropout(sent_embed)

        if "1" in self.args.task_list:
            shift_task_res, shift_task_loss = self.shift_task(sent_embed[:, 0, :], data_item["shift_label"])
        else:
            shift_task_res, shift_task_loss = torch.tensor([0]*bz).to(self.device), torch.tensor(0).to(self.device)

        if "2" in self.args.task_list:
            discriminator_task_res, discriminator_task_loss = self.discriminator_task(
                sent_embed, data_item["input_len_list"], data_item["discriminator_input_index"], bz, col_num, cell_len)
        else:
            discriminator_task_res, discriminator_task_loss = (torch.tensor(0),torch.tensor(0)),torch.tensor(0)

        discriminator_task_loss = self.args.discriminator_loss_weight*discriminator_task_loss
        shift_task_loss = self.args.shift_loss_weight*shift_task_loss
        loss = discriminator_task_loss+shift_task_loss
        return loss, shift_task_res, shift_task_loss.data.cpu(), discriminator_task_res, discriminator_task_loss.data.cpu()

    def discriminator_task(self, sent_embed, input_len_list, discriminator_label, bz, col_num, cell_len):
        sent_embed = sent_embed.reshape(bz, col_num, cell_len, -1)[:,1:-1,:,:]  # size = (bz, col_num, cell_len, 1024)
        new_sent_embed = []
        for sent_embed_item, input_len in zip(sent_embed, input_len_list):
            temp_s = []
            for cell_embed, cell_len in zip(sent_embed_item, input_len):
                temp_s.append(torch.mean(cell_embed[:cell_len, :], dim=0))
            new_sent_embed.append(torch.stack(temp_s))
        sent_embed = torch.stack(new_sent_embed)  # size = (bz, col_num, 1024)

        right_row_index = [(i,) for i in range(bz)]
        target_embed = sent_embed[right_row_index, discriminator_label].permute(1,0,2)
        
        pos_embed_index = self._get_pos_embed_index(col_num, discriminator_label)  # size = (pos_num, bz, col_num-1)
        # pending for optimization
        all_pos_embed = []
        for each_pos_all in pos_embed_index:
            batch_tmp = []
            for index, each_pos_batch in enumerate(each_pos_all):
                batch_tmp.append(sent_embed[index, each_pos_batch, :, ])
            all_pos_embed.append(torch.stack(batch_tmp))
        pos_embed = torch.stack(all_pos_embed)  # size = (pos_num, bz, 1024), pos_num =  col_num*discriminator_ratio
        
        # pending for optimization
        mismatch_row_index = self._get_mismatch_row_index(bz)   # size = bz
        neg_embed = []
        for pos_index, discriminator_label_batch in enumerate(discriminator_label.permute(1,0)):  # size discriminator_label.permute(1,0) = (pos_num,bz) (5,8)
            batch_embed = []
            for batch_index, discriminator_label_each in zip(mismatch_row_index, discriminator_label_batch):
                col_index_list = [i for i in range(0, col_num-2)]
                col_index_list.remove(int(discriminator_label_each.data))
                batch_embed.append(sent_embed[batch_index][col_index_list])
            neg_embed.append(torch.stack(batch_embed))
        neg_embed = torch.stack(neg_embed)

        pos_embed = pos_embed.permute(2,0,1,3)
        neg_embed = neg_embed.permute(2,0,1,3)
        discriminator_task_res_pos_list= []
        discriminator_task_res_neg_list = []
        discriminator_task_loss = 0
        for pos_embed_item, neg_embed_item in zip(pos_embed, neg_embed):
            discriminator_task_res_pos = torch.nn.Sigmoid()(self.discriminator_task_linear_layer(target_embed, pos_embed_item))
            discriminator_task_res_neg = torch.nn.Sigmoid()(self.discriminator_task_linear_layer(target_embed, neg_embed_item))

            discriminator_task_res_pos_list.append(discriminator_task_res_pos)
            discriminator_task_res_neg_list.append(discriminator_task_res_neg)

            discriminator_task_loss += self.discriminator_task_loss(discriminator_task_res_pos, torch.ones(discriminator_task_res_pos.size()).to(self.device) )
            discriminator_task_loss += self.discriminator_task_loss(discriminator_task_res_neg, torch.zeros(discriminator_task_res_neg.size()).to(self.device))

        discriminator_task_loss = discriminator_task_loss / pos_embed.size(0) / 2 
        return (torch.stack(discriminator_task_res_pos_list), torch.stack(discriminator_task_res_neg_list)), discriminator_task_loss

    def shift_task(self, sent_embed, shift_label):
        shift_task_logit = self.shift_task_linear_layer(sent_embed)
        shift_task_res = torch.argmax(shift_task_logit, dim=-1)
        shift_task_loss = self.shift_task_loss(shift_task_logit, shift_label)

        return shift_task_res, shift_task_loss

    def _get_pos_embed_index(self, col_num, discriminator_label):
        pos_embed_index = []
        for discriminator_label_sent in discriminator_label.tolist():
            tmp_2 = []
            for discriminator_label_item in discriminator_label_sent:
                tmp_3 = [i for i in range(0, col_num-2)]
                tmp_3.remove(discriminator_label_item)
                tmp_2.append(tmp_3)
            pos_embed_index.append(torch.tensor(tmp_2))
        pos_embed_index = torch.stack(pos_embed_index)    
        return pos_embed_index.to(self.device).permute(1,0,2)

    def _get_mismatch_row_index(self, bz):
        tmp_shift = random.randint(1, bz-1)
        row_index_2 = [(i,) for i in range(bz)]
        mismatch_row_index =  row_index_2[tmp_shift:] + row_index_2[:tmp_shift]
        return mismatch_row_index 
    
    def get_emb(self, data_item, col_embed):
        input_sent = data_item["sent"]
        bz, col_num, cell_len = input_sent.size()
        attention_mask = (input_sent != self.tokenizer.pad_token_id).bool().reshape(bz,-1).to(self.device) 

        # sent_embed = self.sentence_encoder(input_sent.reshape(bz,-1), attention_mask=attention_mask).last_hidden_state


        sent_embed = self.sentence_encoder.embeddings(input_sent.reshape(bz,-1)).reshape(bz, col_num, cell_len, -1)
        sent_embed = sent_embed+col_embed
        sent_embed = self.sentence_encoder(inputs_embeds=sent_embed.reshape(bz, col_num*cell_len, -1), attention_mask=attention_mask).last_hidden_state.reshape(bz, col_num, cell_len, -1)

        bos_embed = sent_embed.reshape(bz, col_num, cell_len, 1024)[:,0,0,:]

        # sent_embed = sent_embed.reshape(bz, col_num, cell_len, -1)[:,1:-1,:,:]  # size = (bz, col_num, cell_len, 1024)
        # new_sent_embed = []
        # for sent_embed_item, input_len in zip(sent_embed, input_len_list):
        #     temp_s = []
        #     for cell_embed, cell_len in zip(sent_embed_item, input_len):
        #         temp_s.append(torch.mean(cell_embed[:cell_len, :], dim=0))
        #     new_sent_embed.append(torch.stack(temp_s))
        # sent_embed = torch.stack(new_sent_embed)

        return bos_embed.detach()
