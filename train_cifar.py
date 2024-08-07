import random
import numpy as np
import numpy.random as npr
import torch
import time
from torchvision import transforms, datasets
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
from vit import ViT
import argparse
from util import lr_scheduler
from PIL import ImageFilter, ImageOps
import torchvision.transforms.functional as TF
import os


def setup():
    parser=argparse.ArgumentParser('Argument Parser')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--batch_size',type=int,default=128)
    parser.add_argument('--batch_size_test',type=int,default=128)
    parser.add_argument('--lr_ini',type=float,default=1e-5)
    parser.add_argument('--lr_min',type=float,default=1e-5)
    parser.add_argument('--lr_base',type=float,default=5e-4) 
    parser.add_argument('--warmup',type=int,default=5, help="Number of warmup epochs")
    parser.add_argument('--decay',type=int,default=480, help="Number of epochs with liear lr decay")
    parser.add_argument('--cuda',type=int,default=0)
    parser.add_argument('--depth',type=int,default=5)
    parser.add_argument('--num_class',type=int,default=10)
    parser.add_argument('--hdim',type=int,default=128)
    parser.add_argument('--num_heads',type=int,default=4)
    parser.add_argument('--sample_size',type=int,default=1)
    parser.add_argument('--jitter',type=float,default=1e-6, help="Avoid numerical error in Cholesky decomposition")
    parser.add_argument('--drop_rate',type=float,default=0.1, help="Dropout rate")
    parser.add_argument('--patch_size',type=int,default=4)
    parser.add_argument('--max_len',type=int,default=64, help= "Number of tokens in sequence")
    parser.add_argument('--keys_len',type=int,default=16, help= "Number of global inducing keys in each head")
    parser.add_argument('--kernel_type',type=str,default='ard')
    parser.add_argument('--flag_sgp',type=bool,default=False, help="True: SGPA-transformer. False: standard (kernel) transformer")
    parser.add_argument('--epochs',type=int,default=500)
    parser.add_argument('--init_model',type=str,default=None)
    parser.add_argument('--output_folder',type=str,default='./models')
  
    args=parser.parse_args()

    return args


def main(args):
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    transform = transforms.Compose([transforms.ToTensor(),\
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),])
    
    if not os.path.exists('./data'):
        os.makedirs('./data')
    
    dataset = CIFAR10(root='./data/', download=True, transform=transform)
    torch.manual_seed(42)
    val_size = 5000
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
        
    torch.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size_test)

    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')
    model = ViT(device=device, depth=args.depth, patch_size=args.patch_size, in_channels=3, max_len=args.max_len, num_class=args.num_class, hdim=args.hdim, num_heads=args.num_heads, 
                sample_size=args.sample_size, jitter=args.jitter, drop_rate=args.drop_rate, keys_len=args.keys_len, kernel_type=args.kernel_type, flag_sgp=args.flag_sgp)
    model.to(device)
    if args.init_model != None:
        model.load_state_dict(torch.load(args.init_model, map_location=torch.device(device)), strict=False)

    log = []
    anneal_kl = 0. 
    for epoch in range(args.epochs):  
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_scheduler(epoch=epoch, warmup_epochs=args.warmup, decay_epochs=args.decay,\
                                                                         initial_lr=args.lr_ini, base_lr=args.lr_base, min_lr=args.lr_min))
        running_loss = 0.0
        start = time.time()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            loss = model.loss(inputs, labels, min(1., anneal_kl))
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
            if i % 100 == 99:   
                end = time.time()
                log_line = 'epoch = {}, i = {}, avg_running_loss = {}, time = {}'.format(epoch+1, i+1, running_loss / 100, end-start)
                print(log_line)
                log.append(log_line + '\n')
                running_loss = 0.0
                start = time.time()
        anneal_kl += (2 / args.epochs)

        if epoch % 10 == 9 or epoch == 0:
            model.eval()
            with torch.no_grad():
                acc_list = []
                nll_list = []
                for i, data in enumerate(val_loader, 0):
                    inputs, labels = data
            
                    inputs = inputs.to(device)
                    labels = labels.to(device)
  
                    acc, nll = model.acc_nll(inputs, labels)
                    acc_list.append(acc)
                    nll_list.append(nll)
                acc_val = np.mean(np.array(acc_list))
                nll_val = np.mean(np.array(nll_list))   
                log_line = 'epoch = {}, acc_val = {}, nll_val = {}'.format(epoch+1, acc_val, nll_val)
                print(log_line)
                log.append(log_line + '\n')
               
                torch.save(model.state_dict(), args.output_folder+'/epoch'+str(epoch+1))
                model.train()
        with open(args.output_folder+'/training.cklog', "a+") as log_file:
            log_file.writelines(log)
            log.clear()
            
    log_line = 'Finished Training'
    print(log_line)
    log.append(log_line+'\n')
    with open(args.output_folder+'/training.cklog', "a+") as log_file:
        log_file.writelines(log)
        log.clear()
    

if __name__ == '__main__':
    args=setup()
    main(args)  
