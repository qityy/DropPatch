import torch
import torch.nn as nn
import math
from einops import rearrange

class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: [Batch Variate Time]
        x = self.value_embedding(x)
        # x: [Batch Variate d_model]
        return self.dropout(x)

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()

        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False  # no update

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return nn.Parameter(self.pe[:, :x.size(1)].repeat(x.shape[0], 1, 1), requires_grad=True).to(x.device)


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, c_in, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = patch_len
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))

        self.value_embedding = TokenEmbedding(patch_len*c_in, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        # x.shape: bs * l * c
        x = self.padding_patch_layer(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        x = torch.flatten(x, 2, 3)  # bs * n * pc
        # Input encoding
        x = self.value_embedding(x)
        return self.dropout(x)

class SimplePatch(nn.Module):
    """
    only patchify, with no embedding
    """
    def __init__(self, patch_len, stride):
        super().__init__()
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
        self.patch_len = patch_len
        self.stride = stride

    def forward(self, x):
        # x: bs * l * c
        c_in = x.shape[-1]
        seq_len = x.shape[1]
        num_patches = (max(seq_len, self.patch_len) - self.patch_len) // self.stride + 1
        tgt_len = self.patch_len + (num_patches - 1) * self.patch_len
        s_begin = seq_len - tgt_len
        x = x[:, s_begin:, :].permute(0, 2, 1)

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = rearrange(x, 'b c n p -> (b c) n p')

        return x, c_in
