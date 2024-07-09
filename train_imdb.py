import numpy as np
import numpy.random as npr
import torch
import time

from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
from transformer import Transformer
import pandas as pd
from util import lr_scheduler
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

# Importing lackages from our NLP-Hugging Package
from transformers import RobertaConfig, RobertaModel, RobertaTokenizerFast, RobertaForSequenceClassification, AutoTokenizer
import argparse
import os

def setup():
    parser=argparse.ArgumentParser('Argument Parser')
    parser.add_argument('--seed1',type=int,default=42)
    parser.add_argument('--seed2',type=int,default=42)
    parser.add_argument('--batch_size',type=int,default=32)
    parser.add_argument('--batch_size_test',type=int,default=200)
    parser.add_argument('--lr_ini',type=float,default=1e-4)
    parser.add_argument('--lr_min',type=float,default=1e-4)
    parser.add_argument('--lr_base',type=float,default=1e-3) # 5e-5 in the uncer-trans paper
    parser.add_argument('--warmup',type=int,default=0)
    parser.add_argument('--decay',type=int,default=19)
    parser.add_argument('--cuda',type=int,default=0)
    parser.add_argument('--depth',type=int,default=2)
    parser.add_argument('--max_len',type=int,default=512)
    parser.add_argument('--embdim',type=int,default=128)
    parser.add_argument('--num_class',type=int,default=2)
    parser.add_argument('--hdim',type=int,default=128)
    parser.add_argument('--num_heads',type=int,default=8)
    parser.add_argument('--sample_size',type=int,default=1)
    parser.add_argument('--jitter',type=float,default=1e-6)
    parser.add_argument('--drop_rate',type=float,default=0.1)
    parser.add_argument('--keys_len',type=int,default=50)
    parser.add_argument('--kernel_type',type=str,default='exponential')
    parser.add_argument('--flag_sgp',type=bool,default=False)
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--init_model',type=str,default=None)
    parser.add_argument('--output_folder',type=str,default='./models')
  
    args=parser.parse_args()

    return args


class Preprocess:
    def __init__(self, df):
        """
        Constructor for the class
        :param df: Input Dataframe to be pre-processed
        """
        self.df = df
        self.encoded_dict = dict()

    def encoding(self, x):
        if x not in self.encoded_dict.keys():
            self.encoded_dict[x] = len(self.encoded_dict)
        return self.encoded_dict[x]

    def processing(self):
        self.df['encoded_polarity'] = self.df['sentiment'].apply(lambda x: self.encoding(x))
        self.df.drop(['sentiment'], axis=1, inplace=True)
        return self.encoded_dict, self.df


# Creating a CustomDataset class that is used to read the updated dataframe and tokenize the text. 
# The class is used in the return_dataloader function

class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        text = str(self.data.review[index])
        text = " ".join(text.split())
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.encoded_polarity[index], dtype=torch.float)
        } 
    
    def __len__(self):
        return self.len


# Creating a function that returns the dataloader based on the dataframe and the specified train and validation batch size. 

def return_dataloader(df, tokenizer, train_batch_size, validation_batch_size, MAX_LEN, train_size=0.7):
    train_dataset=df.sample(frac=train_size,random_state=42)
    val_dataset=df.drop(train_dataset.index).reset_index(drop=True)[:5000]
    train_dataset = train_dataset.reset_index(drop=True)

    print("FULL Dataset: {}".format(df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("VAL Dataset: {}".format(val_dataset.shape))

    training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
    validation_set = CustomDataset(val_dataset, tokenizer, MAX_LEN)

    train_params = {'batch_size': train_batch_size,
                'shuffle': True,
                'num_workers': 1
                }

    val_params = {'batch_size': validation_batch_size,
                    'shuffle': True,
                    'num_workers': 1
                    }

    training_loader = DataLoader(training_set, **train_params)
    validation_loader = DataLoader(validation_set, **val_params)
    
    return training_loader, validation_loader


def main(args):   
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    df = pd.read_csv('../IMDB_Dataset.csv', encoding='latin-1')
    pre = Preprocess(df)
    encoding_dict, df = pre.processing()

    # Creating the training and validation dataloader using the functions defined above
    train_loader, dev_loader = return_dataloader(df, tokenizer, args.batch_size, 100, args.max_len)

    torch.manual_seed(args.seed2)
    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')
    model = Transformer(device=device, vocab_size=tokenizer.vocab_size, depth=args.depth, max_len=args.max_len, embdim=args.embdim,\
            num_class=args.num_class, hdim=args.hdim, num_heads=args.num_heads, sample_size=args.sample_size, jitter=args.jitter,\
            drop_rate=args.drop_rate, keys_len=args.keys_len, kernel_type=args.kernel_type, flag_sgp=args.flag_sgp)
    model.to(device)

    if args.init_model != None:
        model.load_state_dict(torch.load(args.init_model, map_location=torch.device(device)), strict=False)

    log = []
    for epoch in range(args.epochs): 
         
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_scheduler(epoch=epoch, warmup_epochs=args.warmup, decay_epochs=args.decay,\
                                                                         initial_lr=args.lr_ini, base_lr=args.lr_base, min_lr=args.lr_min))
        running_loss = 0.0
        start = time.time()
        
        for count,data in enumerate(train_loader, 0):
            optimizer.zero_grad()
        
            input_data = data['ids'].to(device, dtype = torch.long)
            input_mask = data['mask'].to(device, dtype = torch.long)
            answers = data['targets'].to(device, dtype = torch.long)
            batch_max_len = torch.max(torch.sum(input_data != 1, 1)).item()

            input_data = input_data[:, :batch_max_len]
            input_mask = input_mask[:, :batch_max_len]

            positional = torch.tile(torch.tensor(np.arange(batch_max_len), dtype=torch.long).unsqueeze(0), (len(answers) ,1)).to(device)
            loss = model.loss(input_data,answers,positional,input_mask)
        
            loss.backward()
            optimizer.step()

            running_loss += loss.item()* len(answers)

            if count % 35 == 34:
                end = time.time()
                log_line = 'epoch = {}, avg_running_loss = {}, time = {}'.format(epoch+1, running_loss/ (len(answers)* 35) , end-start)
                print(log_line)
                log.append(log_line + '\n')
                running_loss = 0.0
                start = time.time()
                with open(args.output_folder+'/training.cklog', "a+") as log_file:
                    log_file.writelines(log)
                    log.clear()

        with torch.no_grad():
            model.eval()
            acc_val, nll_val = model.acc_nll(dev_loader)
            
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
    
