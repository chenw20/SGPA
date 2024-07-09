import numpy.random as npr
import torch
import torch.nn as nn
from mlp import Patch_embedding, ClassficationHead_vit, FC
from sgp_layer import SGP_LAYER


class ViT(torch.nn.Module):
    def __init__(self, device, depth, patch_size, in_channels, max_len, num_class, hdim, num_heads, sample_size, jitter, drop_rate, keys_len, kernel_type, flag_sgp, inference_mode=False):
        super(ViT, self).__init__()
        self.hdim = hdim
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.in_channels = in_channels
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
        self.inference_mode = inference_mode

        self.patch_embedding = Patch_embedding(patch_size=patch_size, in_channels=in_channels, hdim=hdim, max_len=max_len, drop_rate=drop_rate)
        
        self.class_head = ClassficationHead(hdim=hdim, num_class=num_class)

        self.device = device

        self.ln = nn.LayerNorm(hdim)

        self.keys = nn.ParameterList([nn.Parameter(torch.tensor(npr.randn(self.num_heads, 1, self.keys_len, self.hdim), dtype=torch.float32)) for i in range(self.depth)])

        self.sgp_layer_list = nn.ModuleList([SGP_LAYER(device=device, num_heads=num_heads, max_len=max_len, hdim=hdim, kernel_type=self.kernel_type, drop_rate=self.drop_rate, \
            keys_len=self.keys_len, sample_size=self.sample_size, jitter=jitter, flag_sgp=flag_sgp, inference_mode=self.inference_mode)])
        self.mlp_layer_list = nn.ModuleList([FC(hdim=hdim, drop_rate=self.drop_rate)])

        for i in range(1, depth):
            self.sgp_layer_list.append(SGP_LAYER(device=device, num_heads=num_heads, max_len=max_len, hdim=hdim,\
                kernel_type=self.kernel_type, drop_rate=self.drop_rate, keys_len=self.keys_len, sample_size=1, jitter=jitter, flag_sgp=flag_sgp, inference_mode=self.inference_mode))
            self.mlp_layer_list.append(FC(hdim=hdim, drop_rate=self.drop_rate))

    def forward(self, X):
        patch_emb_ln, patch_emb = self.patch_embedding.forward(X) 
        z, total_kl = self.sgp_layer_list[0].forward(patch_emb_ln, self.keys[0])
        
        z_prime = patch_emb.unsqueeze(1) + z 
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
            z, kl = self.sgp_layer_list[i].forward(z_ln, cur_k)
            if self.flag_sgp and not self.inference_mode:
                total_kl += kl
            z_prime = z_prev.unsqueeze(1) + z  
            z_ln = self.ln(z_prime)  
            z = self.mlp_layer_list[i].forward(z_ln) + z_prime  
            if self.flag_sgp and i < self.depth-1:
                cur_k = self.mlp_layer_list[i].forward(self.keys[i+1]) + self.keys[i+1] 
            
        logits = self.class_head.forward(z).squeeze(1) 
        return logits, total_kl
    
    def loss(self, X, y, anneal_kl=1.):
        logits, total_kl = self.forward(X)
        ce_loss = nn.CrossEntropyLoss()
        y = torch.unsqueeze(y,1)
        y = torch.tile(torch.unsqueeze(y, 1), (1, self.sample_size, 1)).view(-1, y.shape[1])
        neg_ElogPyGf = ce_loss(logits.view(-1, self.num_class), y.view(-1))
        if self.flag_sgp and total_kl.item() > 0:
            loss = neg_ElogPyGf + anneal_kl* total_kl
        else:
            loss = neg_ElogPyGf
        return loss
    
    def acc_nll(self, X, y):
        if self.flag_sgp:
            num_samples = 10
        else:
            num_samples = 1
        logits = torch.stack([self.forward(X)[0] for _ in range(num_samples)]) 
        pred_probs = torch.mean(torch.softmax(logits, -1),0) 
        _, pred_hard = torch.max(pred_probs, -1) 

        y = torch.unsqueeze(y,1)
        y = torch.tile(torch.unsqueeze(y, 1), (1, self.sample_size, 1)).view(-1, y.shape[1]) 
        acc = torch.sum(pred_hard.view(-1, 1) == y).item()/ y.shape[0]

        nll_loss = nn.NLLLoss()
        nll=nll_loss(torch.log(pred_probs).view(-1, self.num_class), y.view(-1)).item()
        return acc, nll
