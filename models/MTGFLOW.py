
#%%
from cgitb import reset
from turtle import forward, shape
from numpy import percentile
import torch.nn as nn
import torch.nn.functional as F
from models.NF import MAF
import torch

def interpolate(tensor, index, target_size, mode = 'nearest', dim = 0):
    print(tensor.shape)
    source_length = tensor.shape[dim]
    if source_length > target_size:
        raise AttributeError('no need to interpolate')
    if dim == -1:
        new_tensor = torch.zeros((*tensor.shape[:-1], target_size),dtype=tensor.dtype, device=tensor.device)
    if dim == 0:
        new_tensor = torch.zeros((target_size, *tensor.shape[1:], ),dtype=tensor.dtype, device=tensor.device)
    scale = target_size // source_length
    reset = target_size % source_length
    # if mode == 'nearest':
    new_index = index
    new_tensor[new_index, :] = tensor
    new_tensor[:new_index[0], :] = tensor[0,:].unsqueeze(0)
    for i in range(source_length-1):
        new_tensor[new_index[i]:new_index[i+1] , :] = tensor[i,:].unsqueeze(0)
    new_tensor[new_index[i+1] :,:] = tensor[i+1,:].unsqueeze(0)
    return new_tensor

class GNN(nn.Module):
    """
    The GNN module applied in GANF
    """
    def __init__(self, input_size, hidden_size):

        super(GNN, self).__init__()
        self.lin_n = nn.Linear(input_size, hidden_size)
        self.lin_r = nn.Linear(input_size, hidden_size, bias=False)
        self.lin_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, h, A):
        ## A: K X K
        ## H: N X K  X L X D
        # print(h.shape, A.shape)
        # h_n = self.lin_n(torch.einsum('nkld,kj->njld',h,A))
        # h_n = self.lin_n(torch.einsum('nkld,kj->njld',h,A))
        # print(h.shape, A.shape)
        h_n = self.lin_n(torch.einsum('nkld,nkj->njld',h,A))
        h_r = self.lin_r(h[:,:,:-1])
        h_n[:,:,1:] += h_r
        h = self.lin_2(F.relu(h_n))

        return h

import math
import torch.nn as nn
import matplotlib.pyplot as plt
def plot_attention(data, i, X_label=None, Y_label=None):
  '''
    Plot the attention model heatmap
    Args:
      data: attn_matrix with shape [ty, tx], cutted before 'PAD'
      X_label: list of size tx, encoder tags
      Y_label: list of size ty, decoder tags
  '''
  fig, ax = plt.subplots(figsize=(20, 8)) # set figure size
  heatmap = ax.pcolor(data, cmap=plt.cm.Blues, alpha=0.9)
  fig.colorbar(heatmap)
  # Set axis labels
  if X_label != None and Y_label != None:
    X_label = [x_label for x_label in X_label]
    Y_label = [y_label for y_label in Y_label]
    
    xticks = range(0,len(X_label))
    ax.set_xticks(xticks, minor=False) # major ticks
    ax.set_xticklabels(X_label, minor = False, rotation=45)   # labels should be 'unicode'
    
    yticks = range(0,len(Y_label))
    ax.set_yticks(yticks, minor=False)
    ax.set_yticklabels(Y_label[::-1], minor = False)   # labels should be 'unicode'
    
    ax.grid(True)
    plt.show()
    plt.savefig('graph/attention{:04d}.jpg'.format(i))



class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self, c):
        super(ScaleDotProductAttention, self).__init__()
        self.w_q = nn.Linear(c, c)
        self.w_k = nn.Linear(c, c)
        self.w_v = nn.Linear(c, c)
        self.softmax = nn.Softmax(dim = 1)
        self.dropout = nn.Dropout(0.2)
        # swat_0.2
    def forward(self, x,mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        shape = x.shape
        x_shape = x.reshape((shape[0],shape[1], -1))
        batch_size, length, c = x_shape.size()
        q = self.w_q(x_shape)
        k = self.w_k(x_shape)
        k_t = k.view(batch_size, c, length)  # transpose
        score = (q @ k_t) / math.sqrt(c)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)

        # 3. pass them softmax to make [0, 1] range
        score = self.dropout(self.softmax(score))



        return score, k


class MTGFLOW(nn.Module):

    def __init__ (self, n_blocks, input_size, hidden_size, n_hidden, window_size, n_sensor, dropout = 0.1, model="MAF", batch_norm=True, use_meta=False, meta_input_dim=3, meta_emb_dim=8, meta_inject='concat', amp_branch=False, amp_branch_hidden=32, amp_n_bands=1):
        super(MTGFLOW, self).__init__()

        self.rnn = nn.LSTM(input_size=input_size,hidden_size=hidden_size,batch_first=True, dropout=dropout)
        self.gcn = GNN(input_size=hidden_size, hidden_size=hidden_size)
        self.use_meta = use_meta
        self.meta_emb_dim = meta_emb_dim
        # 작업 G-1: dual-branch disentangle. shape branch(기존 flow NLL)와 별개로,
        # shape 정보만으로 만든 h_shape로 조건부 진폭 p(a|h_shape)=N(μ,σ²)를 학습하는 amplitude head.
        # amp_head 출력 = (μ, logσ). shape encoder(rnn/gcn)와 gradient를 공유(Joint 학습).
        # 작업 G-3a: 진폭 target을 스칼라 → 고정 K-band log-RMS 벡터로 확장. head는 K개 band의
        # (μ_k, logσ_k)를 출력(=2*K). K=1이면 G-1과 동일 구조.
        self.amp_branch = amp_branch
        self.amp_n_bands = int(amp_n_bands)
        if self.amp_branch:
            self.amp_head = nn.Sequential(
                nn.Linear(hidden_size, amp_branch_hidden),
                nn.ReLU(),
                nn.Linear(amp_branch_hidden, 2 * self.amp_n_bands),
            )
        else:
            self.amp_head = None
        # meta 주입 방식: 'concat'(기존, condition C에 C_op를 이어붙임) 또는
        #               'film'(C_op로 C를 곱·덧셈 변조; MAF 입력 차원은 no-meta와 동일).
        self.meta_inject = meta_inject if use_meta else 'concat'
        if self.use_meta:
            self.meta_encoder = nn.Sequential(
                nn.Linear(meta_input_dim, 16),
                nn.ReLU(),
                nn.Linear(16, meta_emb_dim),
            )
            # FiLM(A안): C_op(meta_emb)에서 작은 MLP로 Δγ, β (각 hidden_size 차원)를 만들어 C를 변조.
            # ★ 항등 초기화: 최종 Linear weight/bias를 0으로 두어 시작 시 Δγ=β=0 → C_mod=C (do-no-harm).
            if self.meta_inject == 'film':
                film_out = nn.Linear(16, 2 * hidden_size)
                nn.init.zeros_(film_out.weight)
                nn.init.zeros_(film_out.bias)
                self.film = nn.Sequential(
                    nn.Linear(meta_emb_dim, 16),
                    nn.ReLU(),
                    film_out,
                )
            else:
                self.film = None
        else:
            self.meta_encoder = None
            self.film = None
        # film 모드는 C를 변조만 하므로 MAF 입력(condition) 차원은 no-meta와 동일(hidden_size).
        cond_dim = hidden_size + meta_emb_dim if (self.use_meta and self.meta_inject == 'concat') else hidden_size
        if model=="MAF":
            # self.nf = MAF(n_blocks, n_sensor, input_size, hidden_size, n_hidden, cond_label_size=hidden_size, batch_norm=batch_norm,activation='tanh', mode = 'zero')
            self.nf = MAF(n_blocks, n_sensor, input_size, hidden_size, n_hidden, cond_label_size=cond_dim, batch_norm=batch_norm,activation='tanh')

        self.attention = ScaleDotProductAttention(window_size*input_size)
    def forward(self, x, meta=None):

        return self.test(x, meta).mean()

    def _append_meta_condition(self, h, meta, full_shape):
        if not self.use_meta:
            return h.reshape((-1, h.shape[3]))
        if meta is None:
            raise ValueError("MTGFLOW was created with use_meta=True, but no metadata tensor was provided.")
        meta = meta.to(device=h.device, dtype=h.dtype)
        meta_emb = self.meta_encoder(meta)
        if self.meta_inject == 'film':
            # C_op에서 Δγ, β 산출 → C_mod = (1+Δγ)⊙C + β. window(N축) 단위로 만들어 K·L·H 격자로 broadcast.
            film = self.film(meta_emb)                        # [N, 2H]
            gamma, beta = film.chunk(2, dim=-1)               # 각 [N, H]
            hdim = h.shape[3]
            gamma = gamma.view(full_shape[0], 1, 1, hdim).expand(full_shape[0], full_shape[1], full_shape[2], hdim)
            beta = beta.view(full_shape[0], 1, 1, hdim).expand(full_shape[0], full_shape[1], full_shape[2], hdim)
            h_mod = (1.0 + gamma) * h + beta
            return h_mod.reshape((-1, hdim))
        meta_emb = meta_emb.view(full_shape[0], 1, 1, self.meta_emb_dim)
        meta_emb = meta_emb.expand(full_shape[0], full_shape[1], full_shape[2], self.meta_emb_dim)
        h_flat = h.reshape((-1, h.shape[3]))
        meta_flat = meta_emb.reshape((-1, self.meta_emb_dim))
        return torch.cat([h_flat, meta_flat], dim=-1)

    def test(self, x, meta=None):
        # x: N X K X L X D 
        full_shape = x.shape
        graph,_ = self.attention(x)
        self.graph = graph
        # reshape: N*K, L, D
        x = x.reshape((x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))
        h,_ = self.rnn(x)

        # resahpe: N, K, L, H
        h = h.reshape((full_shape[0], full_shape[1], h.shape[1], h.shape[2]))
        h = self.gcn(h, graph)

        # reshappe N*K*L,H
        h = self._append_meta_condition(h, meta, full_shape)
        x = x.reshape((-1,full_shape[3]))
        log_prob = self.nf.log_prob(x, full_shape[1], full_shape[2], h).reshape([full_shape[0],-1])#
        log_prob = log_prob.mean(dim=1)

        return log_prob

    def forward_disentangle(self, x, meta, a):
        # 작업 G-1: shape branch(log_prob)와 amplitude branch(p(a|h_shape))를 한 번의 encode로 계산.
        # 반환: (shape_logprob[N], amp_logprob[N]) — 둘 다 값이 클수록 정상(로그우도).
        # x: N X K X L X D, a: N X K_band (window별 band별 z-scored log-RMS = batch[4]; G-1은 K_band=1)
        if self.amp_head is None:
            raise ValueError("forward_disentangle requires amp_branch=True.")
        full_shape = x.shape
        graph, _ = self.attention(x)
        self.graph = graph
        x = x.reshape((x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        h, _ = self.rnn(x)
        h = h.reshape((full_shape[0], full_shape[1], h.shape[1], h.shape[2]))
        h = self.gcn(h, graph)  # N X K X L X H

        # amplitude branch: 진폭 정보가 차단된 h(shape encoder 출력)를 (K,L)로 pooling → h_shape.
        # h_shape로 조건부 진폭 정상분포 N(μ, σ²)를 예측. Gaussian NLL(대칭) 사용.
        # 작업 G-3a: head가 K_band개 band의 (μ_k, logσ_k)를 출력(2*K_band). band별 대각 Gaussian
        # log-density를 band축 평균(1/K Σ)해 스칼라 amp_logprob으로 집계 → K_band=1이면 G-1과 동일.
        Kb = self.amp_n_bands
        h_shape = h.mean(dim=(1, 2))                      # N X H
        amp_params = self.amp_head(h_shape)               # N X (2*Kb)
        mu, log_sigma = amp_params[:, :Kb], amp_params[:, Kb:2 * Kb]  # 각 N X Kb
        log_sigma = torch.clamp(log_sigma, min=-7.0, max=7.0)
        a = a.to(device=mu.device, dtype=mu.dtype).reshape(full_shape[0], Kb)
        amp_logprob_band = (-0.5 * ((a - mu) / torch.exp(log_sigma)) ** 2
                            - log_sigma - 0.5 * math.log(2 * math.pi))  # N X Kb
        amp_logprob = amp_logprob_band.mean(dim=1)        # N (band축 평균)

        # shape branch: 기존 test()와 동일한 flow log_prob.
        h = self._append_meta_condition(h, meta, full_shape)
        x = x.reshape((-1, full_shape[3]))
        shape_logprob = self.nf.log_prob(x, full_shape[1], full_shape[2], h).reshape([full_shape[0], -1])
        shape_logprob = shape_logprob.mean(dim=1)         # N

        return shape_logprob, amp_logprob

    def get_graph(self):
        return self.graph

    def locate(self, x, meta=None):
        # x: N X K X L X D 
        full_shape = x.shape

        graph, _ = self.attention(x)
        # reshape: N*K, L, D
        self.graph = graph
        x = x.reshape((x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))
        h,_ = self.rnn(x)

        # resahpe: N, K, L, H
        h = h.reshape((full_shape[0], full_shape[1], h.shape[1], h.shape[2]))
        h = self.gcn(h, graph)

        # reshappe N*K*L,H
        h = self._append_meta_condition(h, meta, full_shape)
        x = x.reshape((-1,full_shape[3]))
        a = self.nf.log_prob(x, full_shape[1], full_shape[2], h)
        log_prob, z = a[0].reshape([full_shape[0],full_shape[1],-1]), a[1].reshape([full_shape[0],full_shape[1],-1])
        


        return log_prob.mean(dim=2), z.reshape((full_shape[0]* full_shape[1],-1))


class test(nn.Module):
    def __init__ (self, n_blocks, input_size, hidden_size, n_hidden, window_size, n_sensor, dropout = 0.1, model="MAF", batch_norm=True, use_meta=False, meta_input_dim=3, meta_emb_dim=8):
        super(test, self).__init__()
        
        if model=="MAF":
            self.nf = MAF(n_blocks, n_sensor, input_size, hidden_size, n_hidden, batch_norm=batch_norm,activation='tanh', mode='zero')
        self.attention = ScaleDotProductAttention(window_size*input_size)
    def forward(self, x, ):
        return self.test(x, ).mean()
    def test(self, x):
        x = x.unsqueeze(2).unsqueeze(3)
        full_shape = x.shape
        x = x.reshape((full_shape[0]*full_shape[1], full_shape[2], full_shape[3]))
        x = x.reshape((-1,full_shape[3]))
        log_prob = self.nf.log_prob(x, full_shape[1], full_shape[2]).reshape([full_shape[0],full_shape[1],-1])#*full_shape[1]*full_shape[2]
        log_prob = log_prob.mean(dim=1)
        return log_prob

    def locate(self, x, meta=None):
        # x: N X K X L X D 
        x = x.unsqueeze(2).unsqueeze(3)
        full_shape = x.shape
  
        x = x.reshape((x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))

        # reshappe N*K*L,H
        x = x.reshape((-1,full_shape[3]))
        a = self.nf.log_prob(x, full_shape[1], full_shape[2])#*full_shape[1]*full_shape[2]
        log_prob, z = a[0].reshape([full_shape[0],full_shape[1],-1]), a[1].reshape([full_shape[0],full_shape[1],-1])
        

        return log_prob.mean(dim=2), z.reshape((full_shape[0]* full_shape[1],-1))


