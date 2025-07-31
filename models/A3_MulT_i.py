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


class A3_MulT_i(nn.Module):
    def __init__(self, hyp_params):
        """Multimodal Transformer with unimodality.
        Args:
            hyp_params (dict/Config): Hyperparameters for model configuration
        """
        super(A3_MulT_i, self).__init__()

        # Convert hyp_params to Config object if it isn't already
        hyp_params = Config(hyp_params)

        # Original dimensions of each modality's features
        self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params.feature_dims

        # Get destination feature dimensions and number of attention heads
        dst_feature_dims, nheads = hyp_params.dst_feature_dim_nheads

        # Set transformed dimensions for all modalities (made equal)
        self.d_l = self.d_a = self.d_v = dst_feature_dims

        # Flags for which modalities to process
        self.vonly = True  # Vision only flag
        self.aonly = False  # Audio only flag
        self.lonly = False  # text only flag

        # Transformer architecture parameters
        self.num_heads = nheads  # Number of attention heads
        self.layers = hyp_params.nlevels  # Number of transformer layers

        # Dropout configurations
        self.attn_dropout = hyp_params.attn_dropout  # Attention dropout
        self.attn_dropout_a = hyp_params.attn_dropout_a  # Audio-specific
        self.attn_dropout_v = hyp_params.attn_dropout_v  # Vision-specific
        self.relu_dropout = hyp_params.relu_dropout  # ReLU dropout
        self.res_dropout = hyp_params.res_dropout  # Residual dropout
        self.out_dropout = hyp_params.output_dropout  # Output dropout
        self.embed_dropout = hyp_params.embed_dropout  # Embedding dropout

        # Attention mask configuration
        self.attn_mask = hyp_params.attn_mask

        # Calculate how many modalities are active (sum of boolean flags)
        self.partial_mode = self.lonly + self.aonly + self.vonly

        # Combined dimension for final projection
        combined_dim = self.d_l * self.partial_mode

        # 1. Temporal convolutional layers for each modality
        # Project raw features to transformer dimension space
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

        # 2. Build transformer networks for each modality pathway
        # Each transformer handles cross-modal attention if needed
        self.forward_l = self.build_transformer(self.lonly, ['la', 'lv'])  # Text pathway
        self.forward_a = self.build_transformer(self.aonly, ['al', 'av'])  # Audio pathway
        self.forward_v = self.build_transformer(self.vonly, ['vl', 'va'])  # Vision pathway

        # 3. Custom concatenation layer for late fusion
        self.concat = CustomConcat([self.lonly, self.aonly, self.vonly], dim=1)

        # 4. Projection layers for final prediction
        self.proj1 = nn.Linear(combined_dim, combined_dim)  # First projection
        self.proj2 = nn.Linear(combined_dim, combined_dim)  # Second projection
        self.out_layer = nn.Linear(combined_dim, 1)  # Final output layer

    def build_transformer(self, only, self_type: list):
        if not only:
            return Identity()

        temp = nn.Sequential(
            Select(index=0),
            self.get_network(self_type[0][0])
        )
        return SequentialWrapper(temp)

    def forward(self, x_l, x_a, x_v, **kwargs):

        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)

        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        res = self.forward_final(proj_x_a, proj_x_v, proj_x_l)

        return res

    def forward_final(self, proj_x_a, proj_x_v, proj_x_l):
        last_h_l = self.forward_l(proj_x_l, proj_x_a, proj_x_v)
        last_h_a = self.forward_a(proj_x_a, proj_x_l, proj_x_v)
        last_h_v = self.forward_v(proj_x_v, proj_x_l, proj_x_a)
        last_hs = self.concat([last_h_l, last_h_a, last_h_v])
        out, last_hs = self.forward_projection(last_hs)
        return {

            'M': out
        }

    def forward_projection(self, last_hs):
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)
        return output, last_hs

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
