import torch


def lr_scheduler(epoch, warmup_epochs, decay_epochs, initial_lr, base_lr, min_lr):
    if epoch <= warmup_epochs:
        pct = epoch / max(warmup_epochs,1)
        return ((base_lr - initial_lr) * pct) + initial_lr
    if epoch > warmup_epochs and epoch < warmup_epochs+decay_epochs:
        pct = 1 - ((epoch - warmup_epochs) / decay_epochs)
        return ((base_lr - min_lr) * pct) + min_lr
    return min_lr


def kernel_ard(X1, X2, log_ls, log_sf):
    X1 = X1 * torch.exp(-log_ls).unsqueeze(1)
    X2 = X2 * torch.exp(-log_ls).unsqueeze(1)
    X1 = X1.permute(0,1,3,2).unsqueeze(4) 
    X2 = X2.unsqueeze(3) 
    return  torch.exp(log_sf).unsqueeze(1) * \
        torch.exp(-0.5* torch.sum((X1-X2.permute(0,1,4,3,2)).pow(2), 2)) 


def kernel_exp(X1, X2, log_ls, log_sf):
    X1 = X1 * torch.exp(-log_ls).unsqueeze(1) 
    X2 = X2 * torch.exp(-log_ls).unsqueeze(1)
    return torch.exp(log_sf).unsqueeze(1)* torch.exp(X1 @ X2.permute(0,1,3,2))