import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, activation="relu", output_attention=False):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout, output_attention)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        new_x, attn = self.attention(x, x, x)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, enc_layers, d_ff, dropout, activation, output_attention,
                 conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout, activation, output_attention)
                                         for i in range(enc_layers)])
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (enc_layer, conv_layer) in enumerate(zip(self.enc_layers, self.conv_layers)):
                delta = delta if i==0 else None
                x, attn = enc_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.enc_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for enc_layer in self.enc_layers:
                x, attn = enc_layer(x)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, activation="relu", output_attention=False):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout, output_attention)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, q, k):
        new_q, attn = self.attention(q, k, k)
        q = q + self.dropout(new_q)

        y = x = self.norm1(q)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(q + y), attn


class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, enc_layers, d_ff, dropout, activation, output_attention,
                 conv_layers=None, norm_layer=None):
        super(Decoder, self).__init__()
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout, activation, output_attention)
                                         for i in range(enc_layers)])
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, k, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (dec_layer, conv_layer) in enumerate(zip(self.dec_layers, self.conv_layers)):
                delta = delta if i==0 else None
                x, attn = dec_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.enc_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for dec_layer in self.dec_layers:
                x, attn = dec_layer(x, k)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, attention_dropout, output_attention):
        super().__init__()
        d_k = d_model // n_heads

        self.n_heads = n_heads
        self.attn = SelfAttention(attention_dropout, output_attention)
        self.W_q = nn.Linear(d_model, d_k * n_heads)
        self.W_k = nn.Linear(d_model, d_k * n_heads)
        self.W_v = nn.Linear(d_model, d_k * n_heads)
        self.out_projection = nn.Linear(d_k * n_heads, d_model)

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.W_q(queries).view(B, L, H, -1)
        keys = self.W_k(keys).view(B, S, H, -1)
        values = self.W_v(values).view(B, S, H, -1)

        out, attn = self.attn(queries, keys, values)
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class SelfAttention(nn.Module):
    def __init__(self, attention_dropout=0.1, output_attention=False):
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.output_attention = output_attention

    def forward(self, queries, keys, values):
        bs, seq_len_q, num_head, d_model = queries.shape
        scale = 1.0 / math.sqrt(d_model)
        scores = torch.einsum('blhd, bshd -> bhls', queries, keys)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum('bhls, bshd-> blhd', A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)



