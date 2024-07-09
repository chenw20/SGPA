import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
from mlp import Embeddings, ClassficationHead, FC
from sgp_layer_mask import SGP_LAYER
                

class Transformer(torch.nn.Module):
    def __init__(self, device, vocab_size, depth, max_len, num_class, embdim, hdim, num_heads, sample_size, jitter, drop_rate, keys_len, kernel_type, flag_sgp, inference_mode=False):
        super(Transformer, self).__init__()
        self.hdim = hdim
        self.max_len = max_len
        self.num_class = num_class
        self.sample_size=sample_size
        self.depth = depth
        self.jitter = jitter
        self.keys_len = keys_len
        self.kernel_type = kernel_type
        self.drop_rate = drop_rate
        self.embdim = embdim
        self.vocab_size = vocab_size
        self.flag_sgp=flag_sgp

        self.embedding = Embeddings(vocab_size=vocab_size,max_len=max_len,emb_size=embdim,h_size=hdim,drop_rate=drop_rate)
        
        self.class_head = ClassficationHead(hdim=hdim, num_class=num_class, drop_rate=drop_rate)

        self.device = device

        self.ln = nn.LayerNorm(hdim)

        self.keys = nn.ParameterList([nn.Parameter(torch.tensor(npr.randn(num_heads, 1, self.keys_len, self.hdim), dtype=torch.float32)) for i in range(self.depth)])

        self.sgp_layer_list = nn.ModuleList([SGP_LAYER(device=device, num_heads=num_heads, hdim=hdim, kernel_type=self.kernel_type, drop_rate=self.drop_rate,\
                 keys_len=self.keys_len, sample_size=self.sample_size, jitter=jitter, flag_sgp=self.flag_sgp, inference_mode=inference_mode)])
        self.mlp_layer_list = nn.ModuleList([FC(hdim=hdim, drop_rate=self.drop_rate)])

        for i in range(1, depth):
            self.sgp_layer_list.append(SGP_LAYER(device=device, num_heads=num_heads, hdim=hdim, kernel_type=self.kernel_type, drop_rate=self.drop_rate,\
                 keys_len=self.keys_len, sample_size=1, jitter=jitter, flag_sgp=self.flag_sgp, inference_mode=inference_mode))
            self.mlp_layer_list.append(FC(hdim=hdim, drop_rate=self.drop_rate))

    def forward(self, input_data,positional, mask):
        emb_ln, emb = self.embedding.forward(input_data, positional)         
        z, total_kl = self.sgp_layer_list[0].forward(emb_ln, self.keys[0], mask) 
        z_prime = emb.unsqueeze(1) + z
        z_ln = self.ln(z_prime) 
        
        z = self.mlp_layer_list[0].forward(z_ln) + z_prime 

        cur_k = None
        if self.flag_sgp:
            cur_k = self.mlp_layer_list[0].forward(self.keys[1]) + self.keys[1] 
        for i in range(1, self.depth):
            z_prev = z.reshape(-1, z.shape[-2], z.shape[-1]) 
            z_ln = self.ln(z_prev)  
            if self.flag_sgp:
              cur_k = self.ln(cur_k) 
            z, kl = self.sgp_layer_list[i].forward(z_ln, cur_k, mask) 
            if total_kl:
                total_kl += kl
            z_prime = z_prev.unsqueeze(1) + z
            z_ln = self.ln(z_prime)  
            z = self.mlp_layer_list[i].forward(z_ln) + z_prime
            if self.flag_sgp and i < self.depth-1:
                cur_k = self.mlp_layer_list[i].forward(self.keys[i+1]) + self.keys[i+1] 
        logits = self.class_head.forward(z, mask).squeeze(1)
        return logits, total_kl 
    
    def loss(self, input_data,answers,positional,input_mask, anneal_kl=1.):
        logits, total_kl = self.forward(input_data,positional,input_mask) 
        ce_loss = nn.CrossEntropyLoss()
        answers = torch.unsqueeze(answers,1) 
        answers = torch.tile(torch.unsqueeze(answers, 1), (1, self.sample_size, 1)).view(-1, answers.shape[1]) 
        neg_ElogPyGf = ce_loss(logits.view(-1, self.num_class), answers.view(-1))
        if total_kl and total_kl.item() > 0:
            loss = neg_ElogPyGf + anneal_kl* total_kl
        else:
            loss = neg_ElogPyGf
        return loss
    
    def acc_nll(self, dev_loader):
        scalar=0
        nll_loss = nn.NLLLoss()
        acc = 0
        dev_size = 0
        for _, data in enumerate(dev_loader, 0):
            input_data = data['ids'].to(self.device, dtype = torch.long)
            input_mask = data['mask'].to(self.device, dtype = torch.long)
            answers = data['targets'].to(self.device, dtype = torch.long)
            dev_size += len(answers)
            
            batch_max_len = torch.max(torch.sum(input_data != 1, 1)).item()
            input_data = input_data[:, :batch_max_len]
            input_mask = input_mask[:, :batch_max_len]
        
            positional = torch.tile(torch.tensor(np.arange(batch_max_len), dtype=torch.long).unsqueeze(0), (len(answers) ,1)).to(self.device)
            
            if self.sample_size == 1: 
                logits = torch.stack([self.forward(input_data,positional,input_mask)[0] for _ in range(10)])
                pred_probs = torch.mean(torch.softmax(logits, -1), 0) 
            else:
                logits, _ = self.forward(input_data,positional,input_mask) 
                logits = logits.reshape(-1, self.sample_size, self.num_class) 
                pred_probs = torch.mean(torch.softmax(logits, -1), 1) 
            _, pred_hard = torch.max(pred_probs, -1) 
            scalar += nll_loss(torch.log(pred_probs).view(-1, self.num_class), answers.view(-1)).item() * len(answers)
            acc += torch.sum(pred_hard.view(-1) == answers.view(-1)).item()
        return acc/ dev_size, scalar/ dev_size
