import numpy.random as npr
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


class Patch_embedding(torch.nn.Module):
    def __init__(self, patch_size, in_channels, hdim, max_len, drop_rate):
        super(Patch_embedding, self).__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.idim = patch_size * patch_size * in_channels
        self.hdim = hdim
        self.max_len = max_len
        self.pos_emb = nn.Parameter(1e-1 * torch.tensor(npr.randn(max_len, hdim), dtype=torch.float32))  

        self.linear_proj = nn.Sequential(
            nn.Conv2d(in_channels, hdim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.ln = nn.LayerNorm(hdim)
        self.dropout = nn.Dropout(drop_rate)
    
    def forward(self, x):  
        input_emb = self.linear_proj(x)
        patch_emb = input_emb + self.pos_emb
        patch_emb = self.dropout(patch_emb)
        return self.ln(patch_emb), patch_emb
        
class Embeddings(torch.nn.Module):
    def __init__(self,vocab_size,max_len,emb_size,h_size, drop_rate):
        super(Embeddings,self).__init__()
        
        self.token_embeds=nn.Embedding(vocab_size,emb_size,padding_idx=0)
        self.pos_embeds=nn.Embedding(max_len,emb_size)
        self.layer_norm=nn.LayerNorm(h_size)
            
        self.project=nn.Linear(emb_size,h_size)
        self.dropout = nn.Dropout(drop_rate)
        
    def forward(self,input_data,pos):
        rep=self.token_embeds(input_data)
        pos=self.pos_embeds(pos)
      
        output=rep+pos
        output=self.project(output)
        output = self.dropout(output)
        
        return self.layer_norm(output), output


class FC(torch.nn.Module):
    def __init__(self, hdim, drop_rate=0.):
        super(FC, self).__init__()
        self.hdim = hdim
        self.act = torch.nn.GELU() 
        self.fc = nn.Sequential(nn.Linear(hdim, hdim), nn.Dropout(drop_rate), self.act, nn.Linear(hdim,hdim), nn.Dropout(drop_rate))
        self.ln = nn.LayerNorm(hdim)

    def forward(self, x):  
        res = self.fc(x)
        return res
    

class ClassficationHead_vit(torch.nn.Module):
    def __init__(self, hdim, num_class):
        super(ClassficationHead_vit, self).__init__()
        self.hdim = hdim
        self.num_class = num_class
        self.fc = nn.Linear(hdim, num_class)
        self.seqpool = nn.Linear(hdim, 1)
        self.ln = nn.LayerNorm(hdim)

    def forward(self, x): 
        # Pooling strategy as in https://arxiv.org/abs/2104.05704 
        res = self.seqpool(x).permute(0,1,3,2) 
        res = torch.softmax(res, -1) 
        res = res @ x 
        res = torch.mean(res, 2) 
        res = self.ln(res)
        res = self.fc(res) 
        return res

class ClassficationHead(torch.nn.Module):
    def __init__(self, hdim, num_class, drop_rate=0.):
        super(ClassficationHead, self).__init__()
        self.hdim = hdim
        self.num_class = num_class
        self.fc = nn.Sequential(nn.Linear(hdim, num_class), nn.Dropout(drop_rate))
        self.ln = nn.LayerNorm(hdim)

    def forward(self, x, input_mask):
        input_mask = input_mask.unsqueeze(-1).unsqueeze(1)
        res = x* input_mask
        res = torch.mean(res, 2)
        res = self.ln(res)
        res = self.fc(res)
        return res
