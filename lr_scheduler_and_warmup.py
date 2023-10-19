import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ConstantLR, ExponentialLR, SequentialLR, CosineAnnealingLR, MultiStepLR
import matplotlib.pyplot as plt
import numpy as np
 
 
def get_lr_at_epoch(cur_epoch, warmup_epoch=5, highlr_epoch=15, base_lr=1e-4, high_lr=5e-4, eta_min=4e-8, T_max=10, warmup_start_lr=1e-5):
    # warm up
    if cur_epoch < warmup_epoch:
        lr_start = warmup_start_lr
        lr_end = high_lr
        alpha = (lr_end - lr_start) / warmup_epoch
        lr = cur_epoch * alpha + lr_start
    elif warmup_epoch <= cur_epoch < highlr_epoch:
        lr = high_lr
    elif cur_epoch == highlr_epoch:
        lr = base_lr
    else:
        lr = cosannealinglr(cur_epoch, base_lr=base_lr, eta_min=eta_min, T_max=T_max)
        
    return lr

def cosannealinglr(cur_epoch, base_lr, eta_min=4e-8, T_max=10):
    lr = eta_min + 0.5 * (base_lr - eta_min) * (1 + np.cos(cur_epoch / T_max * np.pi))
    return lr

def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


if __name__ == "__main__":
    ################################ this is testing mine ################################
    all_lr = []
    start_epoch = 16
    for i in range(start_epoch, 300):
    #     all_lr.append(cosannealinglr(i, 1e-4))
        all_lr.append(get_lr_at_epoch(i))
     
    plt.plot(all_lr)

    
  
    ################################ this is using pytorch builtin lr_scheduler ################################
    class NET(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(45,10)
            self.layer2 = nn.Conv2d(10,50,3)
        def forward(self,x):
            return x
        
    net = NET()


    opt = torch.optim.Adam(net.parameters(), lr=0.0001, betas=(0.843, 0.999), weight_decay=1e-3) # scrach
    scheduler1 = MultiStepLR(opt, milestones=[0], gamma=5, last_epoch=start_epoch-1)
    scheduler2 = CosineAnnealingLR(opt, T_max=10, eta_min=4e-8, last_epoch=start_epoch-1)
    scheduler = SequentialLR(opt, schedulers=[scheduler1, scheduler2], milestones=[20], last_epoch=start_epoch-1)

    import pytorch_warmup as warmup  
    warmup_scheduler = warmup.LinearWarmup(opt, warmup_period=35) #, last_step=min(35, max(0, 7*(start_epoch-1)))) # 35 = 7 * 5，5 epoch warm up，one epoch 7 iteration cuz bs = 704

    all_lr = []
    for i in range(300):
        for j in range(7):
            cur_lr = opt.param_groups[0]['lr']
            all_lr.append(cur_lr)
            print(cur_lr)
            with warmup_scheduler.dampening():   
                pass
            
        with warmup_scheduler.dampening():   
            scheduler.step()
        
    #     print(i,scheduler.get_last_lr())
        
        
    plt.plot(all_lr)
