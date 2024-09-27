import torch
import torch.nn as nn
import torch.nn.functional as func

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DotProductAttention(nn.Module):
    def __init__(self):
        super(DotProductAttention, self).__init__()
    
    def forward(self,Q,K,V,dk,mask=None):
        scaling_factor = 1 / torch.sqrt(torch.tensor(dk,dtype=torch.float32))
        dotProd = (torch.matmul(Q, torch.transpose(K, -2, -1)) / scaling_factor).to(device)

        if mask is not None:
            dotProd += (mask * -math.inf())
        
        sm = fun.softmax(dotProd, dim=-1)
        out = torch.matmul(sm,V)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        print(self.wq.weight)
        print(self.wk.weight)
        print(self.wv.weight)
        self.dense = nn.Linear(d_model, d_model) 

    def split_heads(self, x):
        x = x.view(x.size(0), x.size(1), self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, x, mask=None):
        Q = self.split_heads(self.wq(x))
        K = self.split_heads(self.wk(x))
        V = self.split_heads(self.wv(x))

        dpa = ScaledDotProductAttention()
        attn_output = dpa.forward(Q, K, V)

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(attn_output.size(0), attn_output.size(1), self.d_model)

        return self.dense(attn_output)

MultiHeadAttention(d_model = 12, num_heads = 4)
