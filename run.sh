nohup python run_pretrain.py --ID only_disr_0 --batch_size 16 --discriminator_ratio 0.3 --discriminator_loss_weight 1 --task_list ["1"] --gpu 0  >/dev/null 2>&1 &
nohup python run_pretrain.py --ID only_disr_1 --batch_size 16 --discriminator_ratio 0.5 --discriminator_loss_weight 1 --task_list ["1"] --gpu 1  >/dev/null 2>&1 &

nohup python run_pretrain.py --ID two_task_0  --batch_size 16 --discriminator_ratio 0.3 --discriminator_loss_weight 1 --task_list "["1", "2"]" --gpu 2  >/dev/null 2>&1 &