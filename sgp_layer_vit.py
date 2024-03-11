import numpy.random as npr
import torch
import torch.nn as nn
from util import kernel_ard, kernel_exp


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
            raise ValueError("The argument 'kernel_type' should be either 'exponential' or 'ard'.")
        
        self.sample_size = sample_size
        self.jitter = jitter
        self.device = device
        self.kernel_type = kernel_type 
        
        self.fc_qk = nn.Linear(self.hdim, self.hdim, bias=False)
        if self.kernel_type == 'scale_dot':
            self.fc_k = nn.Linear(self.hdim, self.hdim, bias=False)
        self.fc_v = nn.Linear(self.hdim, self.hdim, bias=False)
        
        self.v = nn.Parameter(torch.tensor(npr.randn(self.num_heads, 1, self.keys_len, self.vdim), dtype=torch.float32))
        self.s_sqrt_ltri = nn.Parameter( torch.tensor(npr.randn(self.num_heads, 1, self.vdim, self.keys_len, self.keys_len), dtype=torch.float32))
        self.log_s_sqrt_diag = nn.Parameter( torch.tensor(npr.randn(self.num_heads, 1, self.vdim, self.keys_len), dtype=torch.float32))
        
        self.W_O = nn.Sequential(nn.Linear(self.hdim, self.hdim), nn.Dropout(self.drop_rate))
    
     
    def get_q_k_v_ssqrt(self, x, cur_k):
        q = self.fc_qk(x).view(x.shape[0], x.shape[1], self.num_heads, self.vdim).permute(0,2,1,3) 
        if self.kernel_type == 'scale_dot':
            k_gamma = self.fc_k(x).view(x.shape[0], x.shape[1], self.num_heads, self.vdim).permute(0,2,1,3)
        else:
            k_gamma = q.clone() 
        v_gamma = self.fc_v(x).view(x.shape[0], x.shape[1], self.num_heads, self.vdim).permute(0,2,1,3) 
        if self.flag_sgp:
            W_qk = self.fc_qk.weight
            k_beta = W_qk.view(self.num_heads, 1, 1, self.vdim, self.hdim) @ cur_k.unsqueeze(-1) 
            k_beta = k_beta.squeeze(-1).permute(1,0,2,3) 
            v_beta = self.v.permute(1,0,2,3)
            log_ssqrt = self.log_s_sqrt_diag.permute(1,0,2,3) 
            return q, k_gamma, k_beta, v_gamma, v_beta, log_ssqrt  
        else:
            return q, k_gamma, v_gamma
        
    def forward(self, x, cur_k):
        if self.flag_sgp:
            q, k_gamma, k_beta, v_gamma, v_beta, log_ssqrt = self.get_q_k_v_ssqrt(x, cur_k)
        else:
            q, k_gamma, v_gamma = self.get_q_k_v_ssqrt(x, cur_k)
            
        if self.kernel_type == 'exponential':
            K_qq = kernel_exp(q, q, self.log_ls_exp, self.log_sf_exp) 
            if self.flag_sgp:
                K_k_beta_k_gamma = kernel_exp(k_beta, k_gamma, self.log_ls_exp, self.log_sf_exp) 
                K_k_beta_k_beta = kernel_exp(k_beta, k_beta, self.log_ls_exp, self.log_sf_exp) 
            K_qk_gamma = K_qq.clone() 
            if self.flag_sgp:    
                K_k_gamma_k_gamma = K_qq.clone()
                K_qk_beta = kernel_exp(q, k_beta, self.log_ls_exp, self.log_sf_exp) 
        elif self.kernel_type == 'ard':
            K_qq = kernel_ard(q, q, self.log_ls_ard, self.log_sf_ard) 
            if self.flag_sgp:
                K_k_beta_k_gamma = kernel_ard(k_beta, k_gamma, self.log_ls_ard, self.log_sf_ard) 
                K_k_beta_k_beta = kernel_ard(k_beta, k_beta, self.log_ls_ard, self.log_sf_ard) 
            K_qk_gamma = K_qq.clone()
            if self.flag_sgp:    
                K_k_gamma_k_gamma = K_qq.clone()
                K_qk_beta = kernel_ard(q, k_beta, self.log_ls_ard, self.log_sf_ard)
        else:
            raise ValueError("The argument 'kernel_type' should be either 'exponential' or 'ard'.")
            
        if self.flag_sgp: 
            mean = (K_qk_gamma - K_qk_beta @ K_k_beta_k_gamma) @ v_gamma + K_qk_beta @ v_beta 
            s_sqrt = torch.exp(log_ssqrt) 
            s_sqrt_diag = torch.diag_embed(s_sqrt) 
            s_sqrt_local = s_sqrt_diag + torch.tril(self.s_sqrt_ltri.permute(1,0,2,3,4), diagonal=-1) 
            
            jitter = self.jitter
            while True:
                try:
                    chol_K_kk = torch.linalg.cholesky(K_k_beta_k_beta + jitter* torch.eye(K_k_beta_k_beta.shape[2]).to(self.device)) 
                    break
                except Exception:
                    jitter = jitter * 10

            covar = K_qq.unsqueeze(2) 
            v1 = torch.triangular_solve(K_qk_beta.permute(0,1,3,2), chol_K_kk, upper=False).solution 
            covar -= v1.unsqueeze(2).permute(0,1,2,4,3) @ v1.unsqueeze(2) 
            v3 = v1.unsqueeze(2).permute(0,1,2,4,3) @ s_sqrt_local 
            covar = covar + v3 @ v3.permute(0,1,2,4,3)   
            
            jitter = self.jitter
            while True:
                try:
                    chol_covar = torch.linalg.cholesky(covar + jitter * torch.eye(covar.shape[3]).to(self.device))  
                    break
                except Exception:
                    jitter = jitter* 10
    
            chol_covar = chol_covar.unsqueeze(2) 
            
            samples = mean.permute(0,1,3,2).unsqueeze(2) + (chol_covar @ \
                torch.randn_like(mean.unsqueeze(2).tile((1,1,self.sample_size,1,1)).unsqueeze(-1))).squeeze(-1)   

            samples = samples.permute(0,1,2,4,3) 
            
            kl = -0.5* self.keys_len* self.vdim * self.num_heads 
            kl += 0.5* torch.mean(torch.sum(s_sqrt_local.pow(2), (-1,-2,-3,-4)))
            
            kl += 0.5* torch.mean(torch.sum(v_beta.permute(0,1,3,2).unsqueeze(3) @ K_k_beta_k_beta.unsqueeze(2) @ v_beta.permute(0,1,3,2).unsqueeze(4), (1,2))) 
            v2 = torch.triangular_solve(K_k_beta_k_gamma, chol_K_kk, upper=False).solution 

            second_term = v2.permute(0,1,3,2) @ v2
            first_term = K_k_gamma_k_gamma
            temp = v_gamma.permute(0,1,3,2).unsqueeze(3) @ (first_term - second_term).unsqueeze(2) @ v_gamma.permute(0,1,3,2).unsqueeze(4)
            kl += 0.5* torch.mean(torch.sum(temp, (1,2)))

            kl -= torch.mean(torch.sum(log_ssqrt, (-1, -2, -3))) 
            
            samples = torch.flatten(samples.permute(0,2,3,1,4),-2,-1) 
            samples = self.W_O(samples)
            return samples, kl
        else: 
            mean = K_qk_gamma @ v_gamma  
            samples = mean.unsqueeze(2) 
            samples = torch.flatten(samples.permute(0,2,3,1,4),-2,-1) 
            samples = self.W_O(samples)
            return samples, None
