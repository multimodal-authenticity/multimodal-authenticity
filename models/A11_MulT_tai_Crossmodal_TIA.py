import torch
from torch import nn
import torch.nn.functional as F
from .modules.transformer import TransformerEncoder


def concat(x: list, dim=0):
    if len(x) == 0:
        return x[0]
    return torch.concat(x, dim=dim)


class Identity(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *x):
        return None


class CustomConcat(nn.Module):
    def __init__(self, mask, dim=0) -> None:
        super().__init__()
        self.dim = dim
        self.mask = mask

    def check(self, i):
        return False if i is None else True

    def forward(self, x: list):
        temp = list(filter(self.check, x))
        return concat(temp, dim=self.dim)


class CrossModalWrapper(nn.Module):
    def __init__(self, trans_a: nn.Module, trans_b, mem: nn.modules) -> None:
        super().__init__()
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.mem = mem

    def forward(self, target_q, kvset_a, kvset_b):
        a = self.trans_a(target_q, kvset_a, kvset_a)
        b = self.trans_a(target_q, kvset_b, kvset_b)
        h = torch.concat([a, b], dim=2)
        h = self.mem(h)
        if type(h) == tuple:
            h = h[0]
        return h[-1]


class SequentialWrapper(nn.Module):
    def __init__(self, seq: nn.Sequential):
        super(SequentialWrapper, self).__init__()
        self.seq = seq

    def forward(self, *inputs):
        h = self.seq(inputs)
        if type(h) == tuple:
            h = h[0]
        return h[-1]


class Select(nn.Module):
    def __init__(self, index) -> None:
        super().__init__()
        self.index = index

    def forward(self, x: list):
        return x[self.index]


class Config:
    def __init__(self, args):
        for k, v in args.items():
            setattr(self, k, v)


class A11_MulT_tai_Crossmodal_TIA(nn.Module):
    def __init__(self, hyp_params):
        """
        Construct a MulT model.
        """
        super(A11_MulT_tai_Crossmodal_TIA, self).__init__()
        hyp_params = Config(hyp_params)
        self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params.feature_dims
        dst_feature_dims, nheads = hyp_params.dst_feature_dim_nheads

        self.d_l = self.d_a = self.d_v = dst_feature_dims

        self.num_heads = nheads
        self.layers = hyp_params.nlevels
        self.attn_dropout = hyp_params.attn_dropout
        self.attn_dropout_a = hyp_params.attn_dropout_a
        self.attn_dropout_v = hyp_params.attn_dropout_v
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.output_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.attn_mask = hyp_params.attn_mask
        combined_dim = self.d_l

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

        self.trans = TransformerEncoder(embed_dim=self.d_l,
                                        num_heads=self.num_heads,
                                        layers=max(self.layers, -1),
                                        attn_dropout=self.attn_dropout,
                                        relu_dropout=self.relu_dropout,
                                        res_dropout=self.res_dropout,
                                        embed_dropout=self.embed_dropout,
                                        attn_mask=self.attn_mask)

        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, 1)

    def forward_projection(self, last_hs):
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)
        return output, last_hs

    def forward_final(self, proj_x_a, proj_x_v, proj_x_l):
        total = torch.concat([proj_x_a, proj_x_v, proj_x_l], dim=0)
        last_hs = self.trans(total)[-1]
        out, last_hs = self.forward_projection(last_hs)
        return {'M': out
                }

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2 * self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2 * self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2 * self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def forward(self, x_l, x_a, x_v, **kwargs):

        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        res = self.forward_final(proj_x_a, proj_x_v, proj_x_l)

        return res
