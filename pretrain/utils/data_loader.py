"""This module defines a configurable RandomShiftTable class."""

import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
import torch.nn
import multiprocessing
from beautifultable import BeautifulTable
from torch.nn.utils.rnn import pad_sequence

def batch_convert_ids_to_tensors(batch_token_ids, ignore_index):
    bz = len(batch_token_ids)
    batch_tensors = [batch_token_ids[i].squeeze(0) for i in range(bz)]
    batch_tensors = pad_sequence(batch_tensors, True, padding_value=ignore_index).long()
    return batch_tensors

class RandomShiftTable(Dataset):
    r"""This module defines a configurable RandomShiftTable class."""

    def __init__(self, tokenizer, args, has_header=True, label_column_index=None):
        r"""Initializes the dataset with given configuration.
        
        Args:
            tokenizer: tokenizer
                Any language model's tokenizer that has encode() method.
            args: str
                Dataset's args.
            has_header: bool
                Whether the data table have a header.
            label_column: int
                Index of label columns.
        """
        with open(args.train_file_path, 'r', encoding='utf-8') as csv_file:
            self.dataframe = pd.read_csv(csv_file, header=0 if has_header else None, dtype=str)
        self.dataframe.fillna(" ", inplace=True)

        # 若有标签列，则预训练不需要
        if label_column_index is not None:
            self.dataframe.drop(self.dataframe.columns[label_column_index], axis=1, inplace=True)

        # 获取列数便于后续的每行分别偏移
        self.num_columns = self.dataframe.shape[1]

        self.max_cell_len = args.max_cell_len
        self.discriminator_num = int(args.discriminator_ratio*self.num_columns)
        self.discriminator_list = list(range(1, self.num_columns-1))
        self.shift_ratio = args.shift_ratio
        self.shift = args.shift if args.shift is not None else self.num_columns-1
        self.last_columns = self.dataframe.columns[-self.shift:]
        self.tokenizer = tokenizer
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id


        print("init data:")
        print("=====================")
        print("max_cell_len", self.max_cell_len)
        print("num_columns", self.num_columns)
        print("shift", self.shift)
        print("shift_ratio", self.shift_ratio)
        print("=====================")
        table = BeautifulTable(maxwidth=200)
        table.column_headers = self.dataframe.head()
        table.append_row(self.dataframe.iloc[1].tolist())
        table.append_row(self.dataframe.iloc[2].tolist())
        print(table)
        print("=====================")



    def __len__(self):
        return self.dataframe.shape[0]


    def __getitem__(self, index):
        X = self.dataframe.iloc[index]
        
        # task 1 : shift_task
        X, shift_y = self.shift_task(X)

        tmp_X_list = []
        input_len_list = []
        for x in X:
            tmp_x = self.tokenizer.encode(x, add_special_tokens=False, return_tensors="pt").squeeze(0)[:self.max_cell_len]
            input_len_list.append(int(tmp_x.size(0)))

            if tmp_x.size(0) < self.max_cell_len:
                pad_x = torch.tensor([self.pad_id]*(self.max_cell_len-tmp_x.size(0)))
                tmp_X_list.append(torch.cat((tmp_x, pad_x), dim=-1))

        X = [torch.tensor([self.bos_id]+[self.pad_id]*(self.max_cell_len-1))]+tmp_X_list+[torch.tensor([self.eos_id]+[self.pad_id]*(self.max_cell_len-1))]

        # task 2 : discriminator_task
        discriminator_y = self.discriminator_task(shift_y)

        return {'sent': torch.stack(X), 'shift_label': torch.tensor([shift_y]).long(), 
        'discriminator_input_index': discriminator_y, "input_len_list":input_len_list}
    

    def shift_task(self, row):
        # 每行各自偏移的列数
        tmp_shift = random.randint(1, self.shift)
        (X, shift_y) = self._transform_row(row, tmp_shift)
        return X, shift_y

    def discriminator_task(self, shift_y):
        # if shift_y ==0:
        #     discriminator_y = torch.tensor(random.sample(self.discriminator_list, self.discriminator_num)).long()
        # else:
        #     discriminator_y = torch.tensor([-1]*self.discriminator_num).long()
        
        discriminator_y = torch.tensor(random.sample(self.discriminator_list, self.discriminator_num)).long()
        return discriminator_y
        

    def _transform_row(self, row, shift=1):
        """Given a row, randomly shift the last k columns to the front.

        Args:
            row: pd.Series
                A row of the dataframe.

        Returns:
            (transformed_row, shift): (list, int)
                The transformed row with label indicating how many col have been shifted.
        """
        last_columns = self.dataframe.columns[-shift:]
        if random.random() < self.shift_ratio:  # shift_ratio的概率进行列变换
            last_columns_data = row[last_columns].tolist()
            transformed_row = last_columns_data + row.drop(last_columns).tolist()
            return (transformed_row, shift)
        else:  # 保持不变
            transformed_row = row.tolist()
            return (transformed_row, 0)



def shift_collate_fn(data):
    batch_data = {'sent': [], 'shift_label': [], 'discriminator_input_index': [], "input_len_list":[]}
    for data_item in data:
        for k, v in batch_data.items():
            tmp = data_item[k]
            batch_data[k].append(tmp)
            
    batch_data['sent'] = torch.stack(batch_data['sent'])   
    batch_data['shift_label'] = torch.stack(batch_data['shift_label']).squeeze()
    batch_data['discriminator_input_index'] = torch.stack(batch_data['discriminator_input_index'])
    batch_data['input_len_list'] = torch.tensor(batch_data['input_len_list'])
    return batch_data


def get_loader(args, tokenizer, num_workers=multiprocessing.cpu_count()):
    dataset = RandomShiftTable(tokenizer, args, has_header=True, label_column_index=-1)
    data_loader = DataLoader(dataset=dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    pin_memory=True,
                                    num_workers=num_workers,
                                    collate_fn=shift_collate_fn,
                                 )
    args.num_columns = dataset.num_columns                             
    return data_loader, args
