import torch

class My_Metrics():
    def __init__(self, ignore_index=0):
        self.ignore_index = ignore_index
    
    def metrics_shift_task_res(self, pred, label):
        pred = torch.cat(pred)
        label = torch.cat(label)
        shift_task_acc = torch.sum(pred == label) / pred.size(0)
        return float(shift_task_acc.data)
    
    def metrics_discriminator_task_res(self, discriminator_task_pos_pred, discriminator_task_neg_pred):
        discriminator_task_pos_pred = torch.cat(discriminator_task_pos_pred).view(-1)
        discriminator_task_neg_pred = torch.cat(discriminator_task_neg_pred).view(-1)
        pos_acc =  torch.sum(discriminator_task_pos_pred>=0.5)/ discriminator_task_pos_pred.size(0)
        neg_acc =  torch.sum(discriminator_task_neg_pred<0.5)/ discriminator_task_neg_pred.size(0)
        discriminator_acc =  float(pos_acc+neg_acc )/2

        pos_mse =  torch.sum(1-discriminator_task_pos_pred)/ discriminator_task_pos_pred.size(0)
        neg_mse =  torch.sum(discriminator_task_neg_pred)/ discriminator_task_neg_pred.size(0)
        discriminator_mse = float(pos_mse+neg_mse )/2
        return discriminator_acc, discriminator_mse