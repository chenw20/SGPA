import torch
import torch.nn as nn


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


class ClassficationHead(torch.nn.Module):
    def __init__(self, hdim, num_class, drop_rate=0.):
        super(ClassficationHead, self).__init__()
        self.hdim = hdim
        self.num_class = num_class
        self.fc = nn.Sequential(nn.Linear(hdim, num_class), nn.Dropout(drop_rate))
        self.seqpool = nn.Linear(hdim, 1, bias=False)
        self.ln = nn.LayerNorm(hdim)
    
    def forward(self, x, input_mask):
        input_mask = input_mask.unsqueeze(-1).unsqueeze(1)
        res = x* input_mask
        res = torch.mean(res, 2)
        res = self.ln(res)
        res = self.fc(res)
        return res
