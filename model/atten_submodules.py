import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention calculation"""

    def __init__(self, num_heads = 2, dropout_rate=0.0, **kwargs):
        """Initialize ScaledDotProductAttention
        Args:
            dropout_rate (float): attention dropout_rate rate
        """
        super().__init__()
        print("inside scaled dot prod attention")

        self.dropout = nn.Dropout(dropout_rate)
        # self.merged_heads = nn.Linear(num_heads, 1)

    def forward(self, Q, K, V, attn_mask = None):

      scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(K.size(-1)) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
      scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
      attn = nn.Softmax(dim=-1)(scores)
      attn = self.dropout(attn)
      context = torch.matmul(attn, V)
      return context, attn



class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=512, num_heads=4, dropout_rate=0.0, attention_type='scaled_dot', query_key_value_weights = [], device = "cuda"):
        super().__init__()
        assert model_dim % num_heads == 0, 'model_dim should be devided by num_heads'
        self.h_size = model_dim
        self.num_heads = num_heads
        self.device = device
        self.head_h_size = model_dim // num_heads

        self.linear_q = nn.Linear(self.h_size, self.h_size)
        self.linear_k = nn.Linear(self.h_size, self.h_size)
        self.linear_v = nn.Linear(self.h_size, self.h_size)
        self.fc0 = nn.Linear(self.h_size, self.h_size)

        self.attention = ScaledDotProductAttention(dropout_rate = 0.2).to(self.device)
        self.dropout = nn.Dropout(dropout_rate)
        self.lnorm = nn.LayerNorm(model_dim)


    def forward(self, q, k, v, attn_mask=None):
        batch_size = q.size(0)

        tensor1 = []
      

        # Residual
        residual = q

        # q, k, v = self.add_positional_mask(q, k, v)

        # Linear projection
        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)

        # Form multi heads
        q = q.view(batch_size, -1, self.num_heads, self.head_h_size).transpose(1,2)  # (h * B, T_q, D / h)
        k = k.view(batch_size, -1, self.num_heads, self.head_h_size).transpose(1,2)  # (h * B, T_k, D / h)
        v = v.view(batch_size, -1, self.num_heads, self.head_h_size).transpose(1,2)  # (h * B, T_v, D / h)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1).to(self.device)  # (h * B, T_q, T_k)

       
        
        context, attention_per_head = self.attention(q, k, v, attn_mask=attn_mask)
        # context: (h * B, T_q, D_v) attention: (h * B, T_q, T_k)

        # Concatenate heads
        context = context.view(batch_size, -1, self.h_size)  # (B, T_q, D)

        
        # Dropout
        output = self.dropout(self.fc0(context))  # (B, T_q, D)

        # Residual connection and Layer Normalization
        output = self.lnorm(residual + output)  # (B, T_q, D)

        return output, attention_per_head

