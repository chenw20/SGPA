import numpy.random as npr
import torch

import torch.nn as nn
from mlp import Embeddings, ClassficationHead, FC
from sgp_layer_trans import SGP_LAYER


class Transformer(torch.nn.Module):
    def __init__(self, device, vocab_size, depth, max_len, embdim, num_class, hdim, num_heads, sample_size, jitter, drop_rate, keys_len, kernel_type, flag_sgp):
        super(Transformer, self).__init__()
        self.hdim = hdim
        self.num_heads = num_heads
        self.max_len = max_len
        self.num_class = num_class
        self.sample_size=sample_size
        self.depth = depth
        self.jitter = jitter
        self.flag_sgp = flag_sgp
        if not self.flag_sgp:
            self.sample_size = 1
        self.keys_len = keys_len
        self.kernel_type = kernel_type
        self.drop_rate = drop_rate
        self.embdim = embdim
        self.vocab_size = vocab_size

        self.embedding = Embeddings(vocab_size=vocab_size,max_len=max_len,emb_size=embdim ,h_size=hdim,drop_rate=drop_rate)
        
        self.class_head = ClassficationHead(hdim=hdim, num_class=num_class, drop_rate=drop_rate)

        self.device = device

        self.ln = nn.LayerNorm(hdim)

        self.keys = nn.ParameterList([nn.Parameter(torch.tensor(npr.randn(self.num_heads,1,self.keys_len,self.embdim+1024), dtype=torch.float32))]+[nn.Parameter(torch.tensor(npr.randn(self.num_heads, 1, self.keys_len, self.hdim), dtype=torch.float32)) for i in range(1,self.depth)])

        self.sgp_layer_list = nn.ModuleList([SGP_LAYER(device=device, num_heads=num_heads, max_len=max_len, hdim=hdim, kernel_type=self.kernel_type, drop_rate=self.drop_rate, keys_len=self.keys_len, sample_size=self.sample_size, jitter=jitter, flag_sgp=flag_sgp)])
        self.mlp_layer_list = nn.ModuleList([FC(hdim=hdim, drop_rate=self.drop_rate)])

        for i in range(1, depth):
            self.sgp_layer_list.append(SGP_LAYER(device=device, num_heads=num_heads, max_len=max_len, hdim=hdim, kernel_type=self.kernel_type, drop_rate=self.drop_rate, keys_len=self.keys_len, sample_size=1, jitter=jitter, flag_sgp=flag_sgp))
            self.mlp_layer_list.append(FC(hdim=hdim, drop_rate=self.drop_rate))

    def forward(self, input_data,positional,input_mask, data):
        emb_ln, emb, keys0 = self.embedding.forward(input_data,positional,self.keys[0], self.device, data) 
        
        z, total_kl = self.sgp_layer_list[0].forward(emb_ln, keys0, input_mask) 
        z_prime = emb.unsqueeze(1) + z 
        z_ln = self.ln(z_prime) 
        
        z = self.mlp_layer_list[0].forward(z_ln) + z_prime 
        if self.depth > 1:
            cur_k = self.mlp_layer_list[0].forward(self.keys[1]) + self.keys[1] 
        for i in range(1, self.depth):
            z_prev = z.reshape(-1, z.shape[-2], z.shape[-1]) 
            z_ln = self.ln(z_prev)  
            cur_k = self.ln(cur_k)
            z, kl = self.sgp_layer_list[i].forward(z_ln, cur_k, input_mask)
            if total_kl:
                total_kl += kl
            z_prime = z_prev.unsqueeze(1) + z  
            z_ln = self.ln(z_prime)  
            z = self.mlp_layer_list[i].forward(z_ln) + z_prime  
            if i < self.depth-1:
                cur_k = self.mlp_layer_list[i].forward(self.keys[i+1]) + self.keys[i+1] 
        logits = self.class_head.forward(z, input_mask).squeeze(1) 
        return logits, total_kl 
    
    def loss(self, input_data,answers,positional,input_mask, data, anneal_kl=1.):
        logits, total_kl = self.forward(input_data,positional,input_mask, data) 
        ce_loss = nn.CrossEntropyLoss()
        answers = answers.unsqueeze(1) 
        answers = answers.unsqueeze(1).tile(1, self.sample_size, 1).view(-1, answers.shape[1]) 
        neg_ElogPyGf = ce_loss(logits.view(-1, self.num_class), answers.view(-1))
        if total_kl:
            loss = neg_ElogPyGf + anneal_kl* total_kl
        else:
            loss = neg_ElogPyGf
        return loss
    
    def pred_nll(self, data_loader):
        nll_sum = 0
        answers_list = []
        pred_hard_list = []
        nll_loss = nn.NLLLoss()
        for i in range(data_loader.num_batches):
            data,input_data,input_mask, positional,answers=data_loader.__load_next__()
            if self.sample_size == 1 and self.flag_sgp: 
                logits = torch.stack([self.forward(input_data,positional,input_mask, data)[0] for _ in range(10)])
                pred_probs = torch.mean(torch.softmax(logits, -1), 0) 
            else:
                logits, _ = self.forward(input_data,positional,input_mask, data) 
                logits = logits.reshape(-1, self.sample_size, self.num_class)
                pred_probs = torch.mean(torch.softmax(logits, -1), 1) 
        
            _, pred_hard = torch.max(pred_probs, -1) 
            answers = answers.unsqueeze(1) #
            pred_hard_list.append(pred_hard)
            answers_list.append(answers)

            nll_sum += len(input_data)* nll_loss(torch.log(pred_probs).view(-1, self.num_class), answers.view(-1)).item()
        
        pred_hard_total = torch.cat(pred_hard_list)
        answers_total = torch.cat(answers_list).squeeze(-1)
        tn = torch.sum((answers_total == 0) * (pred_hard_total == 0)).item()
        tp = torch.sum((answers_total == 1) * (pred_hard_total == 1)).item()
        fp = torch.sum((answers_total == 0) * (pred_hard_total == 1)).item()
        fn = torch.sum((answers_total == 1) * (pred_hard_total == 0)).item()

        mcc=tp*tn-fp*fn
        acc=(tp+tn)/(tp+tn+fp+fn)
        den=(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
        if den==0:
            den=1
        mcc=mcc/den**0.5
        return nll_sum/ len(answers_total), mcc* 100, acc* 100