import torch
from torch import nn
from layers.Embed import PositionalEmbedding, SimplePatch, TokenEmbedding
from layers.Transformer_EncDec import Encoder
from einops import rearrange
import numpy as np
import random


class FlattenHead(nn.Module):
    def __init__(self, head_in, pred_len, dropout=0.1):
        super().__init__()
        self.fltten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(head_in, pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x.shape: bs * c * n * d
        x = self.fltten(x)  # bs * c * (n*d)
        x = self.linear(x)
        x = self.dropout(x)
        return x.permute(0, 2, 1)


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mode = args.mode
        self.enc = Encoder(args.d_model, args.n_heads, args.enc_layers, args.d_ff,
                           args.dropout, args.activation, args.output_attention, norm_layer=nn.LayerNorm(args.d_model))
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.patch_len = args.patch_len
        self.stride = self.patch_len
        self.d_model = args.d_model
        self.discard_ratio = args.mask_ratio
        self.mask_ratio = args.pos_mask_ratio
        self.do_patch = SimplePatch(self.patch_len, self.stride)
        self.patch_embedding = TokenEmbedding(self.patch_len, self.d_model)
        self.dropout = nn.Dropout(args.dropout)
        self.position_embedding = PositionalEmbedding(self.d_model)
        self.output_attention = args.output_attention
        self.num_patches = (max(self.seq_len, self.patch_len)-self.patch_len) // self.stride + 1

        if self.mode == 'pt' or self.mode == 'eval' or self.mode == 'large':
            self.rec_head = nn.Sequential(nn.Dropout(args.dropout), nn.Linear(self.d_model, self.patch_len))
        elif self.mode == 'ft' or self.mode == 'randinit' or self.mode == 'lp' or self.mode == 'fewshot' or self.mode == 'hidden_rep' or self.mode == 'zeroshot':
            pred_head_in = self.num_patches * self.d_model
            self.pred_head = FlattenHead(pred_head_in, self.pred_len, args.head_dropout)

    @staticmethod
    def random_mask(x, mask_ratio):
        # x.shape bs * n * d_model
        B, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand((B, L), device=x.device)

        index_shuffle = torch.argsort(noise, dim=1)  # bs * l
        index_shuffle_back = torch.argsort(index_shuffle, dim=1)

        index_keep = index_shuffle[:, :len_keep]  # bs * len_keep
        index_remove = index_shuffle[:, len_keep:]

        mask = torch.ones((B, L), device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=index_shuffle_back).bool()  # bs * l

        return mask, index_shuffle_back, index_keep, index_remove

    @staticmethod
    def random_shuffle(x):
        B, L, D = x.shape
        noise = torch.rand((B, L), device=x.device)

        index_shuffle = torch.argsort(noise, dim=1)
        index_shuffle_order = torch.argsort(index_shuffle, dim=1)

        x_shuffle = torch.gather(x, dim=1, index=index_shuffle.unsqueeze(-1).repeat(1, 1, D))

        return x_shuffle, index_shuffle_order

    def get_visualization_output(self, rec, mask, index_remove, index_shuffle_back, mask2, c_in):
        # print(f'mask2 sum 1: {mask2.sum()}')
        n_unvis = index_remove.shape[-1]
        padding = torch.zeros((rec.shape[0], n_unvis, self.patch_len), device=rec.device)

        mask_trans = rearrange(mask.unsqueeze(-1).repeat(1, 1, self.patch_len), '(b c) n p -> b (n p) c', c=c_in)

        rec = rearrange(rec, '(b c) n p -> b (n p) c', c=c_in)

        rec_len = (self.num_patches - n_unvis) * self.patch_len
        rec = rec * (self.stdev[:, 0, :].unsqueeze(1).repeat(1, rec_len, 1))
        rec = rec + (self.means[:, 0, :].unsqueeze(1).repeat(1, rec_len, 1))

        rec, _ = self.do_patch(rec)

        rec = torch.gather(torch.cat((rec, padding), dim=1), dim=1, index=index_shuffle_back.unsqueeze(-1).repeat(1, 1, self.patch_len))
        rec_trans = rearrange(rec, '(b c) n p -> b (n p) c', c=c_in)

        mask2 = torch.gather(torch.cat((mask2.unsqueeze(-1).repeat(1, 1, self.patch_len), padding), dim=1), dim=1, index=index_shuffle_back.unsqueeze(-1).repeat(1, 1, self.patch_len))
        mask2_trans = rearrange(mask2, '(b c) n p -> b (n p) c', c=c_in)

        model_visual_outs = {'rec': rec_trans,
                             'mask': mask_trans,
                             'mask2': mask2_trans
                             }
        return model_visual_outs

    def get_visualization_output_eval(self, rec, mask, index_remove, index_shuffle_back, mask2, c_in):
        # print(f'mask2 sum 1: {mask2.sum()}')
        mask_trans = rearrange(mask.unsqueeze(-1).repeat(1, 1, self.patch_len), '(b c) n p -> b (n p) c', c=c_in)

        rec = rearrange(rec, '(b c) n p -> b (n p) c', c=c_in)

        rec_len = self.num_patches * self.patch_len
        rec = rec * (self.stdev[:, 0, :].unsqueeze(1).repeat(1, rec_len, 1))
        rec = rec + (self.means[:, 0, :].unsqueeze(1).repeat(1, rec_len, 1))

        rec, _ = self.do_patch(rec)

        rec_trans = rearrange(rec, '(b c) n p -> b (n p) c', c=c_in)

        model_visual_outs = {'rec': rec_trans,
                             'mask': mask_trans,
                             }
        return model_visual_outs

    def forward_pt(self, x):
        self.means = x.mean(1, keepdim=True).detach()
        x = x - self.means
        self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= self.stdev

        x_patch, c_in = self.do_patch(x)  # (bs*c) * n * p

        mask, index_shuffle_back, index_keep, index_remove = self.random_mask(x_patch, self.discard_ratio)
        x_patch_vis = torch.gather(x_patch, dim=1, index=index_keep.unsqueeze(-1).repeat(1, 1, self.patch_len))  # bs * n_vis * p

        mask2, index_shuffle_back2, index_keep2, index_remove2 = self.random_mask(x_patch_vis, self.mask_ratio)
        x_patch_vis_vis = torch.gather(x_patch_vis, dim=1, index=index_keep2.unsqueeze(-1).repeat(1, 1, self.patch_len))  # bs * n_vis_vis * p

        mask_token = torch.zeros((x_patch_vis_vis.shape[0], index_remove2.shape[1], self.patch_len), requires_grad=False,
                                 device=x_patch_vis.device)

        x_patch_in = torch.cat([x_patch_vis_vis, mask_token], dim=1)  # bs * n_vis * p
        x_patch_in = torch.gather(x_patch_in, dim=1,
                                  index=index_shuffle_back2.unsqueeze(-1).repeat(1, 1, self.patch_len))

        x_patch_embd = self.patch_embedding(x_patch_in)

        x_patch_pe = self.position_embedding(x_patch)
        x_patch_pe_vis = torch.gather(x_patch_pe, dim=1, index=index_keep.unsqueeze(-1).repeat(1, 1, self.d_model))

        enc_in = self.dropout(x_patch_embd + x_patch_pe_vis)  # bs * n_vis * d

        # encoder
        enc_out, attn = self.enc(enc_in)  # bs * n_vis * d

        # reconstruction head
        rec = self.rec_head(enc_out)  # bs * n_vis * p

        loss = (rec - x_patch_vis.detach()) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask2).sum() / mask2.sum()

        model_visual_outs = {}

        # return loss & acc
        return loss, model_visual_outs


    def forward_ft(self, x):
        """
        unshuffle input seq
        """
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev

        x_patch, c_in = self.do_patch(x)
        x_patch = self.patch_embedding(x_patch)  # (bs*c) * n * d
        x_patch_pe = self.position_embedding(x_patch)  # (bs*c) * n * d_model

        enc_in = self.dropout(x_patch + x_patch_pe)

        enc_out, attn = self.enc(enc_in)

        enc_out = rearrange(enc_out, '(b c) n d -> b c n d', c=c_in)

        pred = self.pred_head(enc_out)  # bs * pred_len * c

        pred = pred * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        pred = pred + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return pred, attn

    def forward_ck(self, x):
        """
        unshuffle input seq
        """
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev

        x_patch, c_in = self.do_patch(x)
        x_patch = self.patch_embedding(x_patch)  # (bs*c) * n * d
        x_patch_pe = self.position_embedding(x_patch)  # (bs*c) * n * d_model

        enc_in = self.dropout(x_patch + x_patch_pe)

        enc_out, _ = self.enc(enc_in)

        return enc_out

    def forward_eval(self, x):
        true = x
        self.means = x.mean(1, keepdim=True).detach()
        x = x - self.means
        self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= self.stdev

        x_patch, c_in = self.do_patch(x)  # (bs*c) * n * p

        mask, index_shuffle_back, index_keep, index_remove = self.random_mask(x_patch, self.mask_ratio)
        x_patch_vis = torch.gather(x_patch, dim=1, index=index_keep.unsqueeze(-1).repeat(1, 1, self.patch_len))  # bs * n_vis * d_model, shuffle

        mask_token = torch.zeros((x_patch_vis.shape[0], index_remove.shape[1], self.patch_len), requires_grad=True,
                                 device=x_patch_vis.device)

        x_patch_in = torch.cat([x_patch_vis, mask_token], dim=1)  # bs * n * p, shuffle
        x_patch_in = torch.gather(x_patch_in, dim=1,
                                  index=index_shuffle_back.unsqueeze(-1).repeat(1, 1, self.patch_len))  # unshuffle
        mask_x = rearrange(x_patch_in, '(b c) n p -> b (n p) c', c=c_in)
        mask_trans = rearrange(mask.unsqueeze(-1).repeat(1, 1, self.patch_len), '(b c) n p -> b (n p) c', c=c_in)

        x_patch_in = self.patch_embedding(x_patch_in)
        x_patch_pe = self.position_embedding(x_patch_in)
        enc_in = self.dropout(x_patch_in + x_patch_pe)

        # encoder
        enc_out, attn = self.enc(enc_in)  # bs * n * d

        # reconstruction head
        rec = self.rec_head(enc_out)  # bs * n * p

        loss = (rec - x_patch.detach()) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()

        mask2 = torch.ones_like(index_keep, device=index_keep.device)
        model_visual_outs = self.get_visualization_output_eval(rec, mask, index_remove, index_shuffle_back, mask2, c_in)

        if self.output_attention:
            model_visual_outs['attn'] = attn
        # model_visual_outs = {}

        # return loss & acc
        return loss, model_visual_outs

    def forward(self, x, test=False):
        if self.mode == 'pt' or self.mode == 'large':
            if test:
                return self.forward_eval(x)
            return self.forward_pt(x)
        elif self.mode == 'ft' or self.mode == 'randinit' or self.mode == 'lp' or self.mode == 'fewshot' or self.mode == 'zeroshot':
            return self.forward_ft(x)
        elif self.mode == 'hidden_rep':
            return self.forward_ck(x)
        elif self.mode == 'eval':
            return self.forward_eval(x)
