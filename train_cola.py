import torch
import time
from data_loader import DataLoader
from transformer import Transformer
from data_loader import get_data,get_vocab
import argparse
from util import lr_scheduler


def setup():
    parser=argparse.ArgumentParser('Argument Parser')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--batch_size',type=int,default=32)
    parser.add_argument('--batch_size_test',type=int,default=32)
    parser.add_argument('--lr_ini',type=float,default=1e-5)
    parser.add_argument('--lr_min',type=float,default=1e-5)
    parser.add_argument('--lr_base',type=float,default=5e-4) 
    parser.add_argument('--warmup',type=int,default=5)
    parser.add_argument('--decay',type=int,default=75)
    parser.add_argument('--cuda',type=int,default=0)
    parser.add_argument('--depth',type=int,default=2)
    parser.add_argument('--max_len',type=int,default=100)
    parser.add_argument('--embdim',type=int,default=128)
    parser.add_argument('--num_class',type=int,default=2)
    parser.add_argument('--hdim',type=int,default=256)
    parser.add_argument('--num_heads',type=int,default=4)
    parser.add_argument('--sample_size',type=int,default=1)
    parser.add_argument('--jitter',type=float,default=1e-7)
    parser.add_argument('--drop_rate',type=float,default=0.1)
    parser.add_argument('--keys_len',type=int,default=5)
    parser.add_argument('--kernel_type',type=str,default='exponential')
    parser.add_argument('--flag_sgp',type=bool,default=True)
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--min_word_count',type=int,default=0)
  
    args=parser.parse_args()

    return args


def main(args):    
    data_train,gold_train,data_test,gold_test,data_ood,gold_ood=\
        get_data(['./data/in_domain_train.tsv','./data/in_domain_dev.tsv'],['./data/out_of_domain_dev.tsv'], args.seed)

    word_to_int, _ = get_vocab(data_train,args.min_word_count)
    vocab_size=len(word_to_int)
    
    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(data_train,gold_train,args.batch_size,word_to_int,device)
    test_loader = DataLoader(data_test,gold_test,args.batch_size_test,word_to_int,device,shuffle=False)
    ood_loader = DataLoader(data_ood,gold_ood,args.batch_size_test,word_to_int,device,shuffle=False)

    model = Transformer(device=device, vocab_size=vocab_size, depth=args.depth, max_len=args.max_len, embdim=args.embdim,\
                num_class=args.num_class, hdim=args.hdim, num_heads=args.num_heads, sample_size=args.sample_size, jitter=args.jitter,\
                drop_rate=args.drop_rate, keys_len=args.keys_len, kernel_type=args.kernel_type, flag_sgp=args.flag_sgp)
    model.to(device)

    log = []
    start = time.time()
    running_loss = 0.

    for epoch in range(args.epochs):  
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_scheduler(epoch=epoch, warmup_epochs=args.warmup, decay_epochs=args.decay,\
                                                                         initial_lr=args.lr_ini, base_lr=args.lr_base, min_lr=args.lr_min))
        
        for i in range(train_loader.num_batches):
            optimizer.zero_grad()
            data,input_data,input_mask, positional,answers=train_loader.__load_next__()
            input_data=input_data.to(device) 
            answers=answers.to(device) 
            positional=positional.to(device) 
            input_mask=input_mask.to(device) 
            loss = model.loss(input_data,answers,positional,input_mask, data, 1.)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()* len(input_data) 
        end = time.time()
        log_line = 'epoch = {}, avg_running_loss = {}, time = {}'.format(epoch+1, running_loss / len(data_train), end-start)
        print(log_line)
        log.append(log_line + '\n')
        running_loss = 0.0
        start = time.time()
    
        if epoch % 10 == 9:
            with torch.no_grad():
                model.eval()
                nll_test, mcc_test, acc_test = model.pred_nll(test_loader)
                log_line = 'epoch = {}, acc_test = {}, mcc_test = {}, nll_test = {}'.format(epoch+1, acc_test, mcc_test, nll_test)
                print(log_line)
                log.append(log_line + '\n')
                
                nll_ood, mcc_ood, acc_ood = model.pred_nll(ood_loader)
                log_line = 'epoch = {}, acc_ood = {}, mcc_ood = {}, nll_ood = {}'.format(epoch+1, acc_ood, mcc_ood, nll_ood)
                print(log_line)
                log.append(log_line + '\n')
                torch.save(model.state_dict(), './ckpt_cola/epoch'+str(epoch+1))
            model.train()
            with open('./logs_cola/training.cklog', "a+") as log_file:
                log_file.writelines(log)
                log.clear()
    
    log_line = 'Finished Training'
    print(log_line)
    log.append(log_line+'\n')
    with open('./logs_cola/training.cklog', "a+") as log_file:
        log_file.writelines(log)
        log.clear()

    
if __name__ == '__main__':
    args=setup()
    main(args)
