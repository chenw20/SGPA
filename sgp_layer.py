import time
import numpy.random as npr
import torch
import torch.nn as nn
from util import kernel_ard, kernel_exp, scale_dot


class SGP_LAYER(nn.Module):
    def __init__(self, device, num_heads, max_len, hdim, kernel_type, sample_size, jitter, keys_len, drop_rate, flag_sgp, inference_mode):
        super(SGP_LAYER, self).__init__()
        self.max_len = max_len
        self.num_heads = num_heads
        self.hdim = hdim
        self.vdim = self.hdim // self.num_heads
        self.dq = self.vdim
        self.flag_sgp = flag_sgp
        self.keys_len = keys_len
        self.drop_rate = drop_rate
        self.K_k_beta_k_beta = None
        self.inference_mode = inference_mode
        self.cache_inverse1 = None
        self.cache_inverse2 = None
        
        if kernel_type == 'exponential':
            self.log_sf = nn.Parameter(-4. + 0.* torch.tensor(npr.randn(self.num_heads,1), dtype=torch.float32)) 
            self.log_ls = nn.Parameter(4. + 1.* torch.tensor(npr.randn(self.num_heads,self.dq), dtype=torch.float32)) 
        elif kernel_type == 'ard':
            self.log_sf = nn.Parameter(0. + 0.* torch.tensor(npr.randn(self.num_heads,1), dtype=torch.float32))
            self.log_ls = nn.Parameter(0. + 1.* torch.tensor(npr.randn(self.num_heads,self.dq), dtype=torch.float32)) 
        elif kernel_type == 'scale_dot':
            pass
        else:
            raise ValueError("The argument 'kernel_type' should be either 'exponential', 'ard', or 'scale_dot'.")
        
        self.sample_size = sample_size
        self.jitter = jitter
        self.device = device
        self.kernel_type = kernel_type 
        
        self.fc_qkv = nn.Linear(self.hdim, 2* self.num_heads* self.vdim, bias=False)
        if self.kernel_type == 'scale_dot':
            self.fc_k = nn.Linear(self.hdim, self.hdim, bias=False)

        if self.flag_sgp:
            self.v = nn.Parameter(torch.tensor(npr.randn(self.num_heads, 1, self.keys_len, self.vdim), dtype=torch.float32))
            self.s_sqrt_ltri = nn.Parameter( torch.tensor(npr.randn(self.num_heads, 1, self.vdim, self.keys_len, self.keys_len), dtype=torch.float32))
            self.log_s_sqrt_diag = nn.Parameter( torch.tensor(npr.randn(self.num_heads, 1, self.vdim, self.keys_len), dtype=torch.float32))
        
        self.W_O = nn.Sequential(nn.Linear(self.hdim, self.hdim), nn.Dropout(self.drop_rate))
      
    def get_q_k_v_ssqrt(self, x, cur_k):
        
        q, v_gamma = self.fc_qkv(x).view(x.shape[0], x.shape[1], self.num_heads, 2* self.vdim).permute(0,2,1,3).chunk(chunks=2, dim=-1)
        if self.kernel_type == 'scale_dot':
            k_gamma = self.fc_k(x).view(x.shape[0], x.shape[1], self.num_heads, self.vdim).permute(0,2,1,3)
        else:
            k_gamma = q
        if self.flag_sgp:
            W_qk = self.fc_qkv.weight[:self.hdim]
            k_beta = W_qk.view(self.num_heads, 1, 1, self.vdim, self.hdim) @ cur_k.unsqueeze(-1) 
            k_beta = k_beta.squeeze(-1).permute(1,0,2,3) 
            v_beta = self.v.permute(1,0,2,3)
            log_ssqrt = self.log_s_sqrt_diag.permute(1,0,2,3) 
            return q, k_gamma, k_beta, v_gamma, v_beta, log_ssqrt  
        else:
            return q, k_gamma, v_gamma
        
    def forward(self, x, cur_k):
        # We set W_q = W_k to maintain a valid symmetric deep kernel, so q = k_gamma below when kernel_type='exponential' or 'ard'.
        # We can use different projection matrices if necessary.
        if self.flag_sgp:
            q, k_gamma, k_beta, v_gamma, v_beta, log_ssqrt = self.get_q_k_v_ssqrt(x, cur_k)
        else:
            q, k_gamma, v_gamma = self.get_q_k_v_ssqrt(x, cur_k)
            
        if self.kernel_type == 'exponential':
            if not self.flag_sgp:
                K_qq = kernel_exp(q, q, self.log_ls, self.log_sf)  # [bs, num_heads, max_len, max_len]
            else:
                K_qq, K_qk_beta = kernel_exp(q, torch.cat([q, k_beta.tile(q.shape[0],1,1,1)], 2), \
                    self.log_ls, self.log_sf).tensor_split([self.max_len,],-1) # [bs, num_heads, max_len, max_len + keys_len]
                K_k_beta_k_gamma = K_qk_beta.permute(0,1,3,2)

                if self.K_k_beta_k_beta != None:
                    K_k_beta_k_beta = self.K_k_beta_k_beta
                else:
                    K_k_beta_k_beta = kernel_exp(k_beta, k_beta, self.log_ls, self.log_sf)
                    if self.inference_mode:
                        self.K_k_beta_k_beta = K_k_beta_k_beta
            K_qk_gamma = K_qq
            if self.flag_sgp:    
                K_k_gamma_k_gamma = K_qq
        elif self.kernel_type == 'ard':
            if not self.flag_sgp:
                K_qq = kernel_ard(q, q, self.log_ls, self.log_sf)  
            else:
                K_qq, K_qk_beta = kernel_ard(q, torch.cat([q, k_beta.tile(q.shape[0],1,1,1)], 2), \
                    self.log_ls, self.log_sf).tensor_split([self.max_len,],-1) 
                K_k_beta_k_gamma = K_qk_beta.permute(0,1,3,2)

                if self.K_k_beta_k_beta != None:
                    K_k_beta_k_beta = self.K_k_beta_k_beta
                else:
                    K_k_beta_k_beta = kernel_ard(k_beta, k_beta, self.log_ls, self.log_sf)
                    if self.inference_mode:
                        self.K_k_beta_k_beta = K_k_beta_k_beta
            K_qk_gamma = K_qq
            if self.flag_sgp:    
                K_k_gamma_k_gamma = K_qq
        elif self.kernel_type == 'scale_dot':
            K_qk_gamma = scale_dot(q, k_gamma)
        else:
            raise ValueError("The argument 'kernel_type' should be either 'exponential', 'ard' or 'scale_dot'.")
        
        if not self.flag_sgp: 
            mean = K_qk_gamma @ v_gamma
            samples = mean.unsqueeze(2) 
            samples = torch.flatten(samples.permute(0,2,3,1,4),-2,-1) 
            samples = self.W_O(samples)
            return samples, None
        else:
            s_sqrt = torch.exp(log_ssqrt) 
            s_sqrt_diag = torch.diag_embed(s_sqrt) 
            s_sqrt_local = s_sqrt_diag + torch.tril(self.s_sqrt_ltri.permute(1,0,2,3,4), diagonal=-1)

            if self.inference_mode and self.cache_inverse1 == None:
                K_kk_inverse = torch.linalg.inv(K_k_beta_k_beta + self.jitter* torch.eye(K_k_beta_k_beta.shape[2], device=self.device))
                self.cache_inverse1 = K_kk_inverse
                K_kk_inverse = K_kk_inverse.unsqueeze(2)
                chol_K_kk = torch.linalg.cholesky(K_k_beta_k_beta + self.jitter* torch.eye(K_k_beta_k_beta.shape[2], device=self.device)).unsqueeze(2)
                self.cache_inverse2 = K_kk_inverse @ chol_K_kk @ s_sqrt_local @ s_sqrt_local.permute(0,1,2,4,3) @ chol_K_kk.permute(0,1,2,4,3) @ K_kk_inverse - K_kk_inverse

            # Notice here we make diagonal approximation of the full covariance to accelerate sampling. 
            # Empirically, it doesn't seem to hurt the performance.
            chol_covar1 = torch.diagonal(K_qq.unsqueeze(2) , dim1=3, dim2=4).permute(0,1,3,2).unsqueeze(2)
            if self.inference_mode:
                # During inference, using cached inverse instead of solving linear systems to speed up.
                mean1 = K_qk_gamma @ v_gamma
                mean = mean1 - K_qk_beta @ (self.cache_inverse1 @ (K_k_beta_k_gamma @ v_gamma)) + K_qk_beta @ v_beta 
                chol_covar = (chol_covar1 + ((K_qk_beta.unsqueeze(2) @ self.cache_inverse2) * K_qk_beta.unsqueeze(2)).sum(-1).permute(0,1,3,2).unsqueeze(2)).pow(0.5)
            else:
                jitter = self.jitter
                while True:
                    try:
                        chol_K_kk = torch.linalg.cholesky(K_k_beta_k_beta + jitter* torch.eye(K_k_beta_k_beta.shape[2], device=self.device))
                        break
                    except Exception:
                        jitter = jitter * 10

                v1 = torch.linalg.solve_triangular(chol_K_kk, K_k_beta_k_gamma, upper=False)
                v2 = torch.linalg.solve_triangular(chol_K_kk, K_k_beta_k_gamma @ v_gamma, upper=False)
                v3 = v1.unsqueeze(2).permute(0,1,2,4,3) @ s_sqrt_local

                mean1 = K_qk_gamma @ v_gamma
                mean = mean1 - v1.permute(0,1,3,2) @ v2 + K_qk_beta @ v_beta 
                
                chol_covar2 = v3.pow(2).sum(-1).permute(0,1,3,2).unsqueeze(2) - \
                    v1.unsqueeze(2).permute(0,1,2,4,3).pow(2).sum(-1).permute(0,1,3,2).unsqueeze(2)
                chol_covar = (chol_covar1 + chol_covar2).pow(0.5)
                
            samples = mean.unsqueeze(2) + chol_covar * torch.randn((mean.shape[0], mean.shape[1], self.sample_size, mean.shape[2], mean.shape[3]), device=self.device)   
            samples = torch.flatten(samples.permute(0,2,3,1,4),-2,-1) 
            samples = self.W_O(samples) 

            if self.inference_mode:
                return samples, None
            else:
                kl = -0.5* self.keys_len* self.vdim * self.num_heads 
                kl += 0.5* torch.mean(torch.sum(s_sqrt_local.pow(2), (-1,-2,-3,-4)))            
                kl += 0.5* torch.mean(torch.sum(v_beta.permute(0,1,3,2).unsqueeze(3) @ K_k_beta_k_beta.unsqueeze(2) @ v_beta.permute(0,1,3,2).unsqueeze(4), (1,2))) 
                second_term = v2.permute(0,1,3,2).unsqueeze(3) @ v2.permute(0,1,3,2).unsqueeze(4)
                temp = v_gamma.permute(0,1,3,2).unsqueeze(3) @ mean1.permute(0,1,3,2).unsqueeze(4) - second_term
                kl += 0.5* torch.mean(torch.sum(temp, (1,2)))
                kl -= torch.mean(torch.sum(log_ssqrt, (-1, -2, -3))) 
                return samples, kl
