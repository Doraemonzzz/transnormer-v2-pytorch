import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .helpers import get_activation_fn, get_norm_fn


class NormLinearAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        num_heads,
        act_fun="elu",
        uv_act_fun="swish",
        norm_type="layernorm",
        causal=False,
    ):
        super().__init__()
        # self.q_proj = nn.Linear(embed_dim, hidden_dim)
        # self.k_proj = nn.Linear(embed_dim, hidden_dim)
        self.qk_proj = nn.Linear(embed_dim, hidden_dim)
        self.v_proj = nn.Linear(embed_dim, hidden_dim)
        self.u_proj = nn.Linear(embed_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, embed_dim)
        self.act = get_activation_fn(act_fun)
        self.uv_act = get_activation_fn(uv_act_fun)
        self.num_heads = num_heads
        # self.norm = get_norm_fn(norm_type)(embed_dim // self.num_heads)
        self.norm = get_norm_fn(norm_type)(hidden_dim)
        self.causal = causal
        
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
        # q = self.q_proj(x)
        q = self.qk_proj(x)
        u = self.u_proj(x)
        # k = self.k_proj(y)
        k = self.qk_proj(y)
        v = self.v_proj(y)
        # uv act
        u = self.uv_act(u)
        v = self.uv_act(v)
        # reshape
        q, k, v = map(lambda x: rearrange(x, '... n (h d) -> ... h n d', h=self.num_heads), [q, k, v])
        # act
        q = self.act(q)
        k = self.act(k)
        # normalize
        # q, k = F.normalize(q), F.normalize(k)
        if self.causal:
            if (attn_mask == None):
                attn_mask = (torch.tril(torch.ones(n, n))).to(q)
            l1 = len(q.shape)
            l2 = len(attn_mask.shape)
            for _ in range(l1 - l2):
                attn_mask = attn_mask.unsqueeze(0)
            energy = torch.einsum('... n d, ... m d -> ... n m', q, k)
            energy = energy * attn_mask
            output = torch.einsum('... n m, ... m d -> ... n d', energy, v)
        else:
            kv = torch.einsum('... n d, ... n e -> ... d e', k, v)
            output = torch.einsum('... n d, ... d e -> ... n e', q, kv)
        # normalize
        # output = self.norm(output)
        # reshape
        output = rearrange(output, '... h n d -> ... n (h d)')
        # normalize
        output = self.norm(output)
        # gate
        output = u * output
        # outproj
        output = self.out_proj(output)

        return output
        
        
        
        