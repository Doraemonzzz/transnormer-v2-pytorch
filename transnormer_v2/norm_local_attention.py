import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from .helpers import get_activation_fn, get_norm_fn
from .rpe import Lrpe, LrpeV2

class NormLocalAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        num_heads,
        act_fun="relu",
        uv_act_fun="swish",
        norm_type="layernorm",
        causal=False,
        use_softmax=True,
        use_lrpe=False,
        lrpe_version=1,
    ):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, hidden_dim)
        self.k_proj = nn.Linear(embed_dim, hidden_dim)
        self.v_proj = nn.Linear(embed_dim, hidden_dim)
        self.u_proj = nn.Linear(embed_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, embed_dim)
        self.act = get_activation_fn(act_fun)
        self.uv_act = get_activation_fn(uv_act_fun)
        self.num_heads = num_heads
        self.head_dim = hidden_dim // self.num_heads
        self.causal = causal
        self.use_softmax = use_softmax
        if not self.use_softmax:
            self.norm = get_norm_fn(norm_type)(hidden_dim)
        self.use_lrpe = use_lrpe
        if self.use_lrpe:
            if lrpe_version == 1:
                self.lrpe = Lrpe(self.head_dim)
            else:
                self.lrpe = LrpeV2(self.num_heads, self.head_dim)
        
    def forward(
        self,
        x,
        y=None,
        attn_mask=None,
    ):
        # x: b n d
        if y == None:
            y = x
        n = x.shape[-2]
        # linear map
        q = self.q_proj(x)
        u = self.u_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)
        # uv act
        u = self.uv_act(u)
        v = self.uv_act(v)
        # reshape
        q, k, v = map(lambda x: rearrange(x, '... n (h d) -> ... h n d', h=self.num_heads), [q, k, v])
        # rpe
        if self.use_lrpe:
            q = self.lrpe(q)
            k = self.lrpe(k)
        energy = torch.einsum('... n d, ... m d -> ... n m', q, k) / np.sqrt(self.head_dim)
        
        if self.causal:
            if (attn_mask == None):
                attn_mask = (torch.tril(torch.ones(n, n))).to(q)
            l1 = len(q.shape)
            l2 = len(attn_mask.shape)
            for _ in range(l1 - l2):
                attn_mask = attn_mask.unsqueeze(0)
            if self.use_softmax:
                energy += attn_mask
        if self.use_softmax:
            energy = F.softmax(energy, dim=-1)
        else:
            energy = self.act(energy)
            if self.causal and (not self.use_softmax):
                energy *= torch.exp(attn_mask)
        output = torch.einsum('... n m, ... m d -> ... n d', energy, v)
        # reshape
        output = rearrange(output, '... h n d -> ... n (h d)')
        if not self.use_softmax:
            # normalize
            output = self.norm(output)
        # gate
        output = u * output
        # outproj
        output = self.out_proj(output)

        return output
