import numpy.random as npr
import torch
import torch.nn as nn
import math
import numpy as np


def kernel_ard(X1, X2, log_ls, log_sf):
    X1 = X1 * torch.exp(-log_ls)
    X2 = X2 * torch.exp(-log_ls)
    X1 = torch.unsqueeze(X1, 2) 
    X2 = torch.unsqueeze(X2.permute(0,2,1), 3) 
    return (torch.exp(log_sf) * torch.exp(-0.5* torch.sum((X2-X1.permute(0,3,2,1)).pow(2), 1))).permute(0,2,1)

def kernel_exp(X1, X2, log_ls, log_sf, log_obs_noise, device, scale=1.):
    X1 = X1 * torch.exp(-log_ls) 
    X2 = X2 * torch.exp(-log_ls)
    if X1.shape[1] == X2.shape[1]:
        max_len = X1.shape[1]
        obs_noise = torch.exp(log_obs_noise)
        flag = True
    else:
        flag = False
    if flag:
        return torch.exp(log_sf)* scale* torch.exp(X1 @ X2.permute(0,2,1)) + obs_noise* torch.eye(max_len).to(device).unsqueeze(0)
    else:
        return torch.exp(log_sf)* scale* torch.exp(X1 @ X2.permute(0,2,1))


class SGP_LAYER(nn.Module):
    def __init__(self, device, num_heads, max_len, hdim, kernel_type, sample_size, jitter, keys_len, drop_rate, flag_sgp):
        super(SGP_LAYER, self).__init__()
        self.max_len = max_len
        self.num_heads = num_heads
        self.hdim = hdim
        self.vdim = self.hdim // self.num_heads
        self.dq = self.vdim
        self.flag_sgp = flag_sgp
        self.keys_len = keys_len
        self.drop_rate = drop_rate
        
        if kernel_type == 'exponential':
            self.log_sf_exp = nn.Parameter(-4. + 0.* torch.tensor(npr.randn(self.num_heads,1), dtype=torch.float32)) 
            self.log_ls_exp = nn.Parameter(4. + 1.* torch.tensor(npr.randn(self.num_heads,self.dq), dtype=torch.float32)) 
        elif kernel_type == 'ard':
            self.log_sf_ard = nn.Parameter(0. + 0.* torch.tensor(npr.randn(self.num_heads,1), dtype=torch.float32)) 
            self.log_ls_ard = nn.Parameter(0. + 1.* torch.tensor(npr.randn(self.num_heads,self.dq), dtype=torch.float32)) 
        else:
            raise NotImplementedError
        
        self.sample_size = sample_size
        self.jitter = jitter
        self.device = device
        self.kernel_type = kernel_type 
        
        self.fc_qk = nn.ModuleList([nn.Linear(self.hdim, self.vdim, bias=False)])
        self.fc_v = nn.ModuleList([nn.Linear(self.hdim, self.vdim, bias=False)])
        
        self.v = nn.Parameter(torch.tensor(npr.randn(self.num_heads, 1, self.keys_len, self.vdim), dtype=torch.float32))
        self.s_sqrt_ltri = nn.Parameter( torch.tensor(npr.randn(self.num_heads, 1, self.vdim, self.keys_len, self.keys_len), dtype=torch.float32))
        self.log_s_sqrt_diag = nn.Parameter( torch.tensor(npr.randn(self.num_heads, 1, self.vdim, self.keys_len), dtype=torch.float32))
        self.log_obs_noise = nn.Parameter(torch.tensor(-4 + 0.1* npr.randn(self.num_heads, 1), dtype=torch.float32)) 

        for i in range(1, self.num_heads):
            self.fc_qk.append(nn.Linear(self.hdim, self.vdim, bias=False))
            self.fc_v.append(nn.Linear(self.hdim, self.vdim, bias=False))
        
        self.W_O = nn.Sequential(nn.Linear(self.hdim, self.hdim), nn.Dropout(self.drop_rate))
    
    def forward(self, x, cur_k, input_mask):
        mask_1=input_mask.unsqueeze(-1).view(input_mask.shape[0],-1, input_mask.shape[1])
        mask_2=input_mask.unsqueeze(1).view(input_mask.shape[0],input_mask.shape[1],-1)
        mask_square = mask_1* mask_2 
        mask_for_covar = (1. - mask_square) * torch.eye(mask_square.shape[1]).to(self.device)
        funs = []
        if self.flag_sgp:
            q, k_gamma, k_beta, v_gamma, v_beta, log_ssqrt = self.get_q_k_v_ssqrt(x, cur_k[0], 0)
            samples, kl_total = self.forward_1head(q, k_gamma, k_beta, v_gamma, v_beta, log_ssqrt, 0, input_mask, mask_1, mask_2, mask_square, mask_for_covar)
            funs.append(samples)
            for i in range(1, self.num_heads):
                q, k_gamma, k_beta, v_gamma, v_beta, log_ssqrt = self.get_q_k_v_ssqrt(x, cur_k[i], i)
                samples, kl = self.forward_1head(q, k_gamma, k_beta, v_gamma, v_beta, log_ssqrt, i, input_mask, mask_1, mask_2, mask_square, mask_for_covar)
                funs.append(samples)
                if kl_total:
                    kl_total += kl
        else:
            q, k_gamma, v_gamma = self.get_q_k_v_ssqrt(x, cur_k[0], 0)
            samples, kl_total = self.forward_1head(q, k_gamma, None, v_gamma, None, None, 0, input_mask, mask_1, mask_2, mask_square, mask_for_covar)
            funs.append(samples)
            for i in range(1, self.num_heads):
                q, k_gamma, v_gamma = self.get_q_k_v_ssqrt(x, cur_k[i], i)
                samples, kl = self.forward_1head(q, k_gamma, None, v_gamma, None, None, i, input_mask, mask_1, mask_2, mask_square, mask_for_covar)
                funs.append(samples)
                if kl_total:
                    kl_total += kl

        res = self.W_O(torch.cat(funs, -1)) 
        return res, kl_total 

    def get_q_k_v_ssqrt(self, x, cur_k_i, head_ind):
        q = self.fc_qk[head_ind](x) 
        k_gamma = q.clone() 
        v_gamma = self.fc_v[head_ind](x) 
        if self.flag_sgp:
            k_beta = self.fc_qk[head_ind](cur_k_i) 
            v_beta = self.v[head_ind] 
            log_ssqrt = self.log_s_sqrt_diag[head_ind]
            return q, k_gamma, k_beta, v_gamma, v_beta, log_ssqrt 
        else:
            return q, k_gamma, v_gamma
        
    def forward_1head(self, q, k_gamma, k_beta, v_gamma, v_beta, log_ssqrt, head_ind, input_mask, mask_1, mask_2, mask_square, mask_for_covar):
        if self.kernel_type == 'exponential':
            K_qq = kernel_exp(q, q, self.log_ls_exp[head_ind], self.log_sf_exp[head_ind], self.log_obs_noise[head_ind], self.device) 
            K_qq = K_qq* mask_square
            if self.flag_sgp:
                K_k_beta_k_gamma = kernel_exp(torch.tile(k_beta, (q.shape[0],1,1)), k_gamma, self.log_ls_exp[head_ind], self.log_sf_exp[head_ind], self.log_obs_noise[head_ind], self.device) 
                K_k_beta_k_beta = kernel_exp(k_beta, k_beta, self.log_ls_exp[head_ind], self.log_sf_exp[head_ind] , self.log_obs_noise[head_ind], self.device) 
            if self.flag_sgp:    
                K_k_gamma_k_gamma = K_qq.clone()
                K_qk_beta = kernel_exp(q, torch.tile(k_beta, (q.shape[0],1,1)), self.log_ls_exp[head_ind], self.log_sf_exp[head_ind], self.log_obs_noise[head_ind], self.device) 
                K_qk_beta = K_qk_beta * mask_2
        elif self.kernel_type == 'ard':
            K_qq = kernel_ard(q, q, self.log_ls_ard[head_ind], self.log_sf_ard[head_ind]) 
            K_qq = K_qq* mask_square
            if self.flag_sgp:
                K_k_beta_k_gamma = kernel_ard(torch.tile(k_beta, (q.shape[0],1,1)), k_gamma, self.log_ls_ard[head_ind], self.log_sf_ard[head_ind]) 
                K_k_beta_k_gamma = K_k_beta_k_gamma* mask_1
                K_k_beta_k_beta = kernel_ard(k_beta, k_beta, self.log_ls_ard[head_ind], self.log_sf_ard[head_ind]) 
            K_qk_gamma = K_qq.clone() 
            if self.flag_sgp:    
                K_k_gamma_k_gamma = K_qq.clone()
                K_qk_beta = kernel_ard(q, torch.tile(k_beta, (q.shape[0],1,1)), self.log_ls_ard[head_ind], self.log_sf_ard[head_ind]) 
                K_qk_beta = K_qk_beta * mask_2
        else:
            raise NotImplementedError
            
        v_gamma = mask_2 * v_gamma
        if self.flag_sgp: 
            mean = (K_qk_gamma - K_qk_beta @ K_k_beta_k_gamma) @ v_gamma + K_qk_beta @ v_beta 

            s_sqrt = torch.exp(log_ssqrt)
            s_sqrt_diag = torch.diag_embed(s_sqrt) 
            s_sqrt_local = s_sqrt_diag + torch.tril(self.s_sqrt_ltri[head_ind], diagonal=-1) 
            

            jitter = self.jitter
            while True:
                try:
                    chol_K_kk = torch.linalg.cholesky(K_k_beta_k_beta + jitter* torch.eye(K_k_beta_k_beta.shape[1]).to(self.device)) 
                    break
                except Exception:
                    jitter = jitter * 10

            covar = torch.unsqueeze(K_qq,1) #
            v1 = torch.triangular_solve(K_qk_beta.permute(0,2,1), chol_K_kk, upper=False).solution 
            covar -= torch.unsqueeze(v1,1).permute(0,1,3,2) @ torch.unsqueeze(v1,1) 
            v3 = torch.unsqueeze(v1,1).permute(0,1,3,2) @ s_sqrt_local 
            covar = torch.tile(covar, (1,v3.shape[1],1,1)) + v3 @ v3.permute(0,1,3,2)   
            covar = covar + torch.tile(mask_for_covar.unsqueeze(1), (1, covar.shape[1], 1,1))
            
            jitter = self.jitter
            while True:
                try:
                    chol_covar = torch.linalg.cholesky(covar + jitter * torch.eye(covar.shape[2]).to(self.device))  
                    break
                except Exception:
                    jitter = jitter* 10
            
            chol_covar = chol_covar * mask_square.unsqueeze(1) 
            chol_covar = chol_covar.unsqueeze(1) 
 
            samples = mean.permute(0,1,3,2).unsqueeze(2) + (chol_covar @ \
                torch.randn_like(mean.unsqueeze(2).tile((1,1,self.sample_size,1,1)).unsqueeze(-1))).squeeze(-1) 

            samples = samples.permute(0,1,3,2) 
            
            kl = -0.5* (self.keys_len)* self.vdim 
            kl += 0.5* torch.mean(torch.sum(s_sqrt_local.pow(2), (-1,-2,-3)))
            
            kl += 0.5* torch.mean(torch.sum(torch.unsqueeze(v_beta.permute(0,2,1), 2) @ torch.unsqueeze(K_k_beta_k_beta,1) @ torch.unsqueeze(v_beta.permute(0,2,1),3), 1)) 
            v2 = torch.triangular_solve(K_k_beta_k_gamma, torch.tile(chol_K_kk, (K_k_beta_k_gamma.shape[0],1,1)), upper=False).solution 
            v2 = v2 * mask_1
            second_term = v2.permute(0,2,1) @ v2 
            first_term = K_k_gamma_k_gamma

            temp = torch.unsqueeze(v_gamma.permute(0,2,1), 2) @ torch.unsqueeze(first_term - second_term,1) @ torch.unsqueeze(v_gamma.permute(0,2,1),3)
            kl += 0.5* torch.mean(torch.sum(temp, 1))

            kl -= torch.mean(torch.sum(log_ssqrt, (-1, -2))) 

            return samples, kl
        else: 
            mean = K_qk_gamma @ v_gamma  
            samples = torch.unsqueeze(mean, 1) 
            return samples, None

