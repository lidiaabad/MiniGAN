'''
In this file I will define the discriminator and generator of each GAN. 
Therefore, later we can train the generator and the 

'''

# IMPORTS 
import torch
import torch.nn as nn
from torch import nn, einsum
import torch.nn.functional as F
from utilsgans import *
from einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np
#from pesq import pesq
import copy
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTM, GRU
from torch.nn.modules.normalization import LayerNorm


class SerializableModule(nn.Module):

    subclasses = {}

    def __init__(self):
        super().__init__()

    @classmethod
    def register_model(cls, model_name):
        def decorator(subclass):
            cls.subclasses[model_name] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, arc):
        if arc not in cls.subclasses:
            raise ValueError('Bad model name {}'.format(arc))

        return cls.subclasses[arc]()

    def save(self, filename):
        torch.save(self.state_dict(), filename +'.pt')

    def save_entire_model(self, filename):
        torch.save(self, filename +'_entire.pt')

    def save_scripted(self, filename):
        scripted_module = torch.jit.script(self)
        scripted_module.save(filename + '.jit')

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))


#-------------------------------------------------Helper functions---------------------------------------------------

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)


class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()



class Scale(nn.Module):
    def __init__(self, scale, fn):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale


def pesq_loss(clean, noisy, sr=16000):
    try:
        pesq_score = pesq(sr, clean, noisy, "wb")
    except:
        # error can happen due to silent period
        pesq_score = -1
    return pesq_score


def batch_pesq(clean, noisy):
    pesq_score = Parallel(n_jobs=-1)(
        delayed(pesq_loss)(c, n) for c, n in zip(clean, noisy)
    )
    pesq_score = np.array(pesq_score)
    if -1 in pesq_score:
        return None
    pesq_score = (pesq_score - 1) / 3.5
    return torch.FloatTensor(pesq_score).to("cuda")

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

def phase_losses(phase_r, phase_g, n_fft):

    dim_freq = n_fft // 2 + 1
    dim_time = phase_r.size(-1)

    gd_matrix = (torch.triu(torch.ones(dim_freq, dim_freq), diagonal=1) - torch.triu(torch.ones(dim_freq, dim_freq), diagonal=2) - torch.eye(dim_freq)).to(phase_g.device)
    gd_r = torch.matmul(phase_r.permute(0, 2, 1), gd_matrix)
    gd_g = torch.matmul(phase_g.permute(0, 2, 1), gd_matrix)

    iaf_matrix = (torch.triu(torch.ones(dim_time, dim_time), diagonal=1) - torch.triu(torch.ones(dim_time, dim_time), diagonal=2) - torch.eye(dim_time)).to(phase_g.device)
    iaf_r = torch.matmul(phase_r, iaf_matrix)
    iaf_g = torch.matmul(phase_g, iaf_matrix)

    ip_loss = torch.mean(anti_wrapping_function(phase_r-phase_g))
    gd_loss = torch.mean(anti_wrapping_function(gd_r-gd_g))
    iaf_loss = torch.mean(anti_wrapping_function(iaf_r-iaf_g))

    return ip_loss, gd_loss, iaf_loss


def anti_wrapping_function(x):

    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)


def pesq_score(utts_r, utts_g, h):

    pesq_score = Parallel(n_jobs=30)(delayed(eval_pesq)(
                            utts_r[i].squeeze().cpu().numpy(),
                            utts_g[i].squeeze().cpu().numpy(), 
                            h.sampling_rate)
                          for i in range(len(utts_r)))
    pesq_score = np.mean(pesq_score)

    return pesq_score


def eval_pesq(clean_utt, esti_utt, sr):
    try:
        pesq_score = pesq(sr, clean_utt, esti_utt)
    except:
        # error can happen due to silent period
        pesq_score = -1

    return pesq_score

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

################################################# General (before also some helper functions)

class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class FeedForwardModule(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
#CMGAN
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, max_pos_emb=512):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.max_pos_emb = max_pos_emb
        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None, context_mask=None):
        n, device, h, max_pos_emb, has_context = (
            x.shape[-2],
            x.device,
            self.heads,
            self.max_pos_emb,
            exists(context),
        )
        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        # shaw's relative positional embedding
        seq = torch.arange(n, device=device)
        dist = rearrange(seq, "i -> i ()") - rearrange(seq, "j -> () j")
        dist = dist.clamp(-max_pos_emb, max_pos_emb) + max_pos_emb
        rel_pos_emb = self.rel_pos_emb(dist).to(q)
        pos_attn = einsum("b h n d, n r d -> b h n r", q, rel_pos_emb) * self.scale
        dots = dots + pos_attn

        if exists(mask) or exists(context_mask):
            mask = default(mask, lambda: torch.ones(*x.shape[:2], device=device))
            context_mask = (
                default(context_mask, mask)
                if not has_context
                else default(
                    context_mask, lambda: torch.ones(*context.shape[:2], device=device)
                )
            )
            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(mask, "b i -> b () i ()") * rearrange(
                context_mask, "b j -> b () () j"
            )
            dots.masked_fill_(~mask, mask_value)

        attn = dots.softmax(dim=-1)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return self.dropout(out)

#MPSENET
class AttentionModule(nn.Module):
    def __init__(self, dim, n_head=8, dropout=0.):
        super(AttentionModule, self).__init__()
        self.attn = nn.MultiheadAttention(dim, n_head, dropout=dropout)
        self.layernorm = nn.LayerNorm(dim)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        x = self.layernorm(x)
        x, _ = self.attn(x, x, x, 
                         attn_mask=attn_mask,
                         key_padding_mask=key_padding_mask)
        return x
    

class ConformerConvModule(nn.Module):
    def __init__(self, dim, expansion_factor=2, kernel_size=31, dropout=0.):
        #careful other conform module with Causal=False as init arg
        super(ConformerConvModule, self).__init__()
        inner_dim = dim * expansion_factor
        #padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)
        self.ccm = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n c -> b c n'),
            nn.Conv1d(dim, inner_dim*2, 1),
            nn.GLU(dim=1),
            nn.Conv1d(inner_dim, inner_dim, kernel_size=kernel_size,
                      padding=get_padding(kernel_size), groups=inner_dim), # DepthWiseConv1d 
            nn.BatchNorm1d(inner_dim),
            nn.SiLU(),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange('b c n -> b n c'),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ccm(x)

class ConvolutionModule(nn.Module):
    def __init__(self, dim, expansion_factor=2, kernel_size=31, dropout=0.):
        #careful other conform module with Causal=False as init arg
        super(ConvolutionModule, self).__init__()
        inner_dim = dim * expansion_factor
        #padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)
        self.ccm = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n c -> b c n'),
            nn.Conv1d(dim, inner_dim*2, 1),
            nn.GLU(dim=1),
            nn.Conv1d(inner_dim, inner_dim, kernel_size=kernel_size,
                      padding=get_padding(kernel_size), groups=inner_dim), # DepthWiseConv1d 
            nn.BatchNorm1d(inner_dim),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange('b c n -> b n c'),
            nn.Dropout(dropout), 
        )

    def forward(self, x):
        return self.ccm(x)


class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super(SPConvTranspose2d, self).__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.0)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1)
        )
        self.r = r

    def forward(self, x):
        x = self.pad1(x)
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out

######################################################### Dense blocks
#for CMGAN, TPTGAN
class DilatedDenseNet(nn.Module):
    def __init__(self, depth=4, in_channels=64):
        super(DilatedDenseNet, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.0)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2**i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(
                self,
                "pad{}".format(i + 1),
                nn.ConstantPad2d((1, 1, pad_length, 0), value=0.0),
            )
            setattr(
                self,
                "conv{}".format(i + 1),
                nn.Conv2d(
                    self.in_channels * (i + 1),
                    self.in_channels,
                    kernel_size=self.kernel_size,
                    dilation=(dil, 1),
                ),
            )
            setattr(
                self,
                "norm{}".format(i + 1),
                nn.InstanceNorm2d(in_channels, affine=True),
            )
            setattr(self, "prelu{}".format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, "pad{}".format(i + 1))(skip)
            out = getattr(self, "conv{}".format(i + 1))(out)
            out = getattr(self, "norm{}".format(i + 1))(out)
            out = getattr(self, "prelu{}".format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out

#for MPSEnet
class DenseBlock(nn.Module):
    def __init__(self, kernel_size=(3, 3), depth=4, dense_channel=64):
        super(DenseBlock, self).__init__()
        self.depth = depth
        self.dense_block = nn.ModuleList([])
        for i in range(depth):
            dil = 2 ** i
            dense_conv = nn.Sequential(
                nn.Conv2d(dense_channel*(i+1), dense_channel, kernel_size, dilation=(dil, 1),
                          padding=get_padding_2d(kernel_size, (dil, 1))),
                nn.InstanceNorm2d(dense_channel, affine=True),
                nn.PReLU(dense_channel)
            )
            self.dense_block.append(dense_conv)

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x

#for TPTGAN. some diferentes with DenseBlock
class DenseBlockTPT(nn.Module):
    def __init__(self, input_size, depth=5, in_channels=64):
        super(DenseBlockTPT, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2 ** i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(self, 'pad{}'.format(i + 1), nn.ConstantPad2d((1, 1, pad_length, 0), value=0.))
            setattr(self, 'conv{}'.format(i + 1),
                    nn.Conv2d(self.in_channels * (i + 1), self.in_channels, kernel_size=self.kernel_size,
                              dilation=(dil, 1)))
            setattr(self, 'norm{}'.format(i + 1), nn.LayerNorm(input_size))
            setattr(self, 'prelu{}'.format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, 'pad{}'.format(i + 1))(skip)
            out = getattr(self, 'conv{}'.format(i + 1))(out)
            out = getattr(self, 'norm{}'.format(i + 1))(out)
            out = getattr(self, 'prelu{}'.format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out


############################################################ Encoders
#uses Conv+Norm+Prelu + DilatedDenseNet (depth=4) + Conv+Norm+PreLU --> CMGAN
class DenseEncoder(nn.Module):
    def __init__(self, in_channel, channels=64):
        super(DenseEncoder, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, channels, (1, 1), (1, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )
        self.dilated_dense = DilatedDenseNet(depth=4, in_channels=channels)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 3), (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.dilated_dense(x)
        x = self.conv_2(x)
        return x

#uses Conv+Norm+Prelu + DenseBlock! (depth=4) + Conv+Norm+PreLU --> MPSENet
class DenseEncoderdense(nn.Module):
    def __init__(self, in_channel, dense_channel=64):
        super(DenseEncoderdense, self).__init__()
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, dense_channel, (1, 1)),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel))

        #careful uses dense block instead of dilateddense
        self.dense_block = DenseBlock(depth=4, dense_channel=dense_channel) # [b, h.dense_channel, ndim_time, h.n_fft//2+1]

        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(dense_channel, dense_channel, (1, 3), (1, 2)),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel))

    def forward(self, x):
        x = self.dense_conv_1(x)  # [b, 64, T, F]
        x = self.dense_block(x)   # [b, 64, T, F]
        x = self.dense_conv_2(x)  # [b, 64, T, F//2]
        return x

#TPTUSES THE DENSEBLOCK DIRECTLY?

#################################################### Conformers and transformers 
#FeedForward (Scale), Attention, ConformerConvModule, FeedForward (Scale), Norm --> CMGAN
class ConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head=64,
        heads=8,
        ff_mult=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        attn_dropout=0.0,
        ff_dropout=0.0,
        conv_dropout=0.0
    ):
        super().__init__()
        self.ff1 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.attn = Attention(
            dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout
        )
        self.conv = ConformerConvModule(
            dim=dim,
            expansion_factor=conv_expansion_factor,
            kernel_size=conv_kernel_size,
            dropout=conv_dropout,
        )
        self.ff2 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)

        self.attn = PreNorm(dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        x = self.ff1(x) + x
        x = self.attn(x, mask=mask) + x
        x = self.conv(x) + x
        x = self.ff2(x) + x
        x = self.post_norm(x)
        return x

#The same as the ConformerBlock but uses AttentionModule --> MPSENet
class ConformerBlockModule(nn.Module):
    def __init__(self, dim, n_head=8, ffm_mult=4, ccm_expansion_factor=2, ccm_kernel_size=31, 
                 ffm_dropout=0., attn_dropout=0., ccm_dropout=0.):
        super(ConformerBlockModule, self).__init__()
        self.ffm1 = FeedForward(dim, ffm_mult, dropout=ffm_dropout)
        self.attn = AttentionModule(dim, n_head, dropout=attn_dropout)
        self.ccm = ConformerConvModule(dim, ccm_expansion_factor, ccm_kernel_size, dropout=ccm_dropout)
        self.ffm2 = FeedForward(dim, ffm_mult, dropout=ffm_dropout)
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + 0.5 * self.ffm1(x)
        x = x + self.attn(x)
        x = x + self.ccm(x)
        x = x + 0.5 * self.ffm2(x)
        x = self.post_norm(x)
        return x

class LitConformer(nn.Module):
    def __init__(self, dim, n_head=8, ffm_mult=4, ccm_expansion_factor=2, ccm_kernel_size=15, 
                 ffm_dropout=0., attn_dropout=0., ccm_dropout=0.):
        super(LitConformer, self).__init__()
        self.ffm1 = FeedForward(dim, ffm_mult, dropout=ffm_dropout)
        #note that the attention moduale has layer norm before the multihead and  drop out after
        self.attn = AttentionModule(dim, n_head, dropout=attn_dropout) 
        self.ccm = ConvolutionModule(dim, ccm_expansion_factor, ccm_kernel_size, dropout=ccm_dropout)
        self.ffm2 = FeedForward(dim, ffm_mult, dropout=ffm_dropout)
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + 0.5 * self.ffm1(x)
        x = x + self.attn(x)
        x = x + self.ccm(x)
        x = x + 0.5 * self.ffm2(x)
        x = self.post_norm(x)
        return x

#multiheadattention + Gru + Bidirectional + Norm --> TPTGAN
class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, bidirectional=True, dropout=0.1, activation="gelu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        # self.linear1 = Linear(d_model, dim_feedforward)
        self.gru = GRU(d_model, d_model*2, 1, bidirectional=bidirectional)
        self.dropout = Dropout(dropout)
        # self.linear2 = Linear(dim_feedforward, d_model)
        if bidirectional:
            self.linear2 = Linear(d_model*2*2, d_model)
        else:
            self.linear2 = Linear(d_model*2, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        self.gru.flatten_parameters()
        out, h_n = self.gru(src)
        del h_n
        src2 = self.linear2(self.dropout(self.activation(out)))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

#lo dejo pero parece que no se utiliza pq se usa el transformer encoder layer.  --> TPTGAN
class ConformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, bidirectional=True, dropout=0.1, activation="gelu"):
        super(ConformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        # self.linear1 = Linear(d_model, dim_feedforward)
        self.gru = GRU(d_model, d_model*2, 1, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.conv = ConformerConvModule(dim=d_model, causal=False, expansion_factor=2,
                                        kernel_size=31)
        # self.linear2 = Linear(dim_feedforward, d_model)
        if bidirectional:
            self.linear2 = Linear(d_model*2*2, d_model)
        else:
            self.linear2 = Linear(d_model*2, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(ConformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.conv(src)
        src = self.norm3(src)
        self.gru.flatten_parameters()
        out, h_n = self.gru(src)
        del h_n
        src2 = self.linear2(self.dropout(self.activation(out)))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

#uses transformer Encoder Layer --> TPTGAN
class Dual_Transformer(nn.Module):
    """
    Deep duaL-path RNN.
    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        dropout: float, dropout ratio. Default is 0.
        num_layers: int, number of stacked RNN layers. Default is 1.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(self, input_size, output_size, dropout=0, num_layers=1):
        super(Dual_Transformer, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.input = nn.Sequential(
            nn.Conv2d(input_size, input_size // 2, kernel_size=1),
            nn.PReLU()
        )

        # dual-path RNN
        self.row_trans = nn.ModuleList([])
        self.col_trans = nn.ModuleList([])
        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])
        for i in range(num_layers):
            self.row_trans.append(
                TransformerEncoderLayer(d_model=input_size // 2, nhead=4, dropout=dropout, bidirectional=True))
            # ConformerEncoderLayer(d_model=input_size // 2, nhead=4, dropout=dropout, bidirectional=True))
            self.col_trans.append(
                TransformerEncoderLayer(d_model=input_size // 2, nhead=4, dropout=dropout, bidirectional=True))
            # ConformerEncoderLayer(d_model=input_size // 2, nhead=4, dropout=dropout, bidirectional=True))
            self.row_norm.append(nn.GroupNorm(1, input_size // 2, eps=1e-8))
            self.col_norm.append(nn.GroupNorm(1, input_size // 2, eps=1e-8))

        # output layer
        self.output = nn.Sequential(nn.PReLU(),
                                    nn.Conv2d(input_size // 2, output_size, 1)
                                    )

    def forward(self, input):
        #  input --- [b,  c,  num_frames, frame_size]  --- [b, c, dim2, dim1]
        b, c, dim2, dim1 = input.shape
        output = self.input(input)
        for i in range(len(self.row_trans)):
            row_input = output.permute(3, 0, 2, 1).contiguous().view(dim1, b * dim2, -1)  # [dim1, b*dim2, c]
            row_output = self.row_trans[i](row_input)  # [dim1, b*dim2, c]
            row_output = row_output.view(dim1, b, dim2, -1).permute(1, 3, 2, 0).contiguous()  # [b, c, dim2, dim1]
            row_output = self.row_norm[i](row_output)  # [b, c, dim2, dim1]
            output = output + row_output  # [b, c, dim2, dim1]

            col_input = output.permute(2, 0, 3, 1).contiguous().view(dim2, b * dim1, -1)  # [dim2, b*dim1, c]
            col_output = self.col_trans[i](col_input)  # [dim2, b*dim1, c]
            col_output = col_output.view(dim2, b, dim1, -1).permute(1, 3, 0, 2).contiguous()  # [b, c, dim2, dim1]
            col_output = self.col_norm[i](col_output)  # [b, c, dim2, dim1]
            output = output + col_output  # [b, c, dim2, dim1]

        del row_input, row_output, col_input, col_output
        output = self.output(output)  # [b, c, dim2, dim1]

        return output

#myblock for TPTGAN

class DPTBlock(nn.Module):
    #def __init__(self, input_size, nHead, dim_feedforward):
    def __init__(self, input_size, nHead):
        super(DPTBlock, self).__init__()

        #d_model, nhead, dim_feedforward, dropout=0, activation="relu"
        self.intra_transformer = TransformerEncoderLayer(d_model=input_size, nhead=nHead, dropout=0)
        self.inter_transformer = TransformerEncoderLayer(d_model=input_size, nhead=nHead, dropout=0)
        
    def forward(self,x):
        
        B, N, K, P = x.shape
        
        # intra DPT
        row_input =  x.permute(0, 3, 2, 1).contiguous().view(B*P, K, N) # [B, N, K, S] -> [B, P, K, N] -> [B*P, K, N]
        row_output = self.intra_transformer(row_input.permute(1, 0, 2).contiguous()).permute(1, 0, 2).contiguous()
        row_output = row_output.view(B, P, K, N).permute(0, 3, 2, 1).contiguous()  # [B*P, K, N] -> [B, P, K, N]
        
        output = x + row_output

        #inter DPT
        col_input = output.permute(0, 2, 3, 1).contiguous().view(B*K, P, N) # [B, P, K, N] -> [B, K, P, N] -> [B*K, P, N]
        col_output = self.inter_transformer(col_input.permute(1, 0, 2).contiguous()).permute(1, 0, 2).contiguous()
        col_output = col_output.view(B, K, P, N).permute(0, 3, 1, 2).contiguous() # [B*K, P, N] -> [B, K, P, N]
        
        output = output + col_output
        
        return output 
    
#uses: ConformerBlock (2: one for time and one for req) --> CMGAN
class TSCB(nn.Module):
    def __init__(self, num_channel=64):
        super(TSCB, self).__init__()
        self.time_conformer = ConformerBlock(
            dim=num_channel,
            dim_head=num_channel // 4,
            heads=4,
            conv_kernel_size=31,
            attn_dropout=0.2,
            ff_dropout=0.2,
        )
        self.freq_conformer = ConformerBlock(
            dim=num_channel,
            dim_head=num_channel // 4,
            heads=4,
            conv_kernel_size=31,
            attn_dropout=0.2,
            ff_dropout=0.2,
        )

    def forward(self, x_in):
        b, c, t, f = x_in.size()
        x_t = x_in.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        x_t = self.time_conformer(x_t) + x_t
        x_f = x_t.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        x_f = self.freq_conformer(x_f) + x_f
        x_f = x_f.view(b, t, f, c).permute(0, 3, 1, 2)
        return x_f

#the same  but uses ConformerBlockModule with other attention --> MPSENet
class TSConformerBlock(nn.Module):
    def __init__(self, dense_channel=64):
        super(TSConformerBlock, self).__init__()
        self.time_conformer = ConformerBlockModule(dim=dense_channel,  n_head=4, ccm_kernel_size=31, 
                                             ffm_dropout=0.2, attn_dropout=0.2)
        self.freq_conformer = ConformerBlockModule(dim=dense_channel,  n_head=4, ccm_kernel_size=31, 
                                             ffm_dropout=0.2, attn_dropout=0.2)

    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        x = self.time_conformer(x) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        x = self.freq_conformer(x) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x
    
class TSLitConformerBlock(nn.Module):
    def __init__(self,dense_channel=64):
        super(TSLitConformerBlock, self).__init__()
        self.time_conformer = LitConformer(dim=dense_channel,  n_head=4, ccm_kernel_size=31, 
                                             ffm_dropout=0.2, attn_dropout=0.2)
        self.freq_conformer = LitConformer(dim=dense_channel,  n_head=4, ccm_kernel_size=31, 
                                             ffm_dropout=0.2, attn_dropout=0.2)

    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        x = self.time_conformer(x) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        x = self.freq_conformer(x) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x


######################################################## Decoders
#uses DilatedDenseNet + SCPConvTranspose + Conv + Norm + PreLu + Conv + PreLU --> CMGAN
class MaskDecoder(nn.Module):
    def __init__(self, num_features, num_channel=64, out_channel=1):
        super(MaskDecoder, self).__init__()
        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
        self.sub_pixel = SPConvTranspose2d(num_channel, num_channel, (1, 3), 2)
        self.conv_1 = nn.Conv2d(num_channel, out_channel, (1, 2))
        self.norm = nn.InstanceNorm2d(out_channel, affine=True)
        self.prelu = nn.PReLU(out_channel)
        self.final_conv = nn.Conv2d(out_channel, out_channel, (1, 1))
        self.prelu_out = nn.PReLU(num_features, init=-0.25)

    def forward(self, x):
        x = self.dense_block(x)
        x = self.sub_pixel(x)
        x = self.conv_1(x)
        x = self.prelu(self.norm(x))
        x = self.final_conv(x).permute(0, 3, 2, 1).squeeze(-1)
        return self.prelu_out(x).permute(0, 2, 1).unsqueeze(1)

#uses DilatedDenseNet + SCPConvTranspose +  Norm + PreLu + Conv  --> GMGAN
class ComplexDecoder(nn.Module):
    def __init__(self, num_channel=64):
        super(ComplexDecoder, self).__init__()
        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
        self.sub_pixel = SPConvTranspose2d(num_channel, num_channel, (1, 3), 2)
        self.prelu = nn.PReLU(num_channel)
        self.norm = nn.InstanceNorm2d(num_channel, affine=True)
        self.conv = nn.Conv2d(num_channel, 2, (1, 2))

    def forward(self, x):
        x = self.dense_block(x)
        x = self.sub_pixel(x)
        x = self.prelu(self.norm(x))
        x = self.conv(x)
        return x

#uses DenseNet! + ConvTranspose +  Conv + Norm `+ PreLU + Conv + Sigmoid--> MPSENet
#difference in ConvTranspose2d and SCPCOnvTranspose of CMCGAN (r of SCPCOnvTranspose, but set to 1 so almost the same)
class MaskDecoderdense(nn.Module):
    def __init__(self, out_channel=1, dense_channel=64):
        super(MaskDecoderdense, self).__init__()
        self.dense_block = DenseBlock(depth=4, dense_channel=dense_channel)
        self.mask_conv = nn.Sequential(
            nn.ConvTranspose2d(dense_channel, dense_channel, (1, 3), (1, 2)),
            nn.Conv2d(dense_channel, out_channel, (1, 1)),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.PReLU(out_channel),
            nn.Conv2d(out_channel, out_channel, (1, 1))
        )
        self.lsigmoid = LearnableSigmoid_2d(512//2+1) #n_fft=512

    def forward(self, x):
        x = self.dense_block(x)
        x = self.mask_conv(x)
        x = x.permute(0, 3, 2, 1).squeeze(-1)
        x = self.lsigmoid(x).permute(0, 2, 1).unsqueeze(1)
        return x

#uses DenseNet! + ConvTranspose + Norm + PreLu + Conv + Conv --> MPSENet
class PhaseDecoder(nn.Module):
    def __init__(self, out_channel=1, dense_channel=64):
        super(PhaseDecoder, self).__init__()
        self.dense_block = DenseBlock(depth=4, dense_channel=dense_channel)
        self.phase_conv = nn.Sequential(
            nn.ConvTranspose2d(dense_channel, dense_channel, (1, 3), (1, 2)),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel)
        )
        self.phase_conv_r = nn.Conv2d(dense_channel, out_channel, (1, 1))
        self.phase_conv_i = nn.Conv2d(dense_channel, out_channel, (1, 1))

    def forward(self, x):
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        return x


################################################################### CMCGAN #########################################################
'''
- Generator
    - Encoder 
        - Conv + Norm + Prelu
        - Dilated DenseNet
        - Conv + Norm + Prelu
    - Latent space
        - TwoStage-Conformers xN
            - Conformer Time
            - Conformer Freq
    - Decoders
        - Magnitude decoder
            - Dilated Dense Net
            - TransposeConv + Conv + Norm + Prelu + Conv (but in code extra prelu at the end??)
        - Complex decoder
            - Dilated Dense Net
            - TransposeConv + Conv + Norm + Prelu + Conv (but in code no first conv??)

    Uses: DenseEncoder + TSCB x4, MaskDecoder, ComplexDecoder
            
- Discriminator
    - (Conv block + Norm + PreLU ) x4: 
    
    Uses no other class, all nn.

'''
#----------------------------------------------------------------DISCRIMINATOR-------------------------------------------------------
@SerializableModule.register_model('CMGAN discriminator')
class CMGANDiscriminator(SerializableModule):
    def __init__(self, ndf, in_channel=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(in_channel, ndf, (3,3), (2, 2), (1, 1), bias=False)
            ),
            nn.InstanceNorm2d(ndf, affine=True),
            nn.PReLU(ndf),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf, ndf * 2, (3,3), (2, 2), (1, 1), bias=False)
            ),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.PReLU(2 * ndf),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * 2, ndf * 4, (3,3), (2, 2), (1, 1), bias=False)
            ),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.PReLU(4 * ndf),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * 4, ndf * 8, (3,3), (2, 2), (1, 1), bias=False)
            ),
            nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.PReLU(8 * ndf),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(ndf * 8, ndf * 4)),
            nn.Dropout(0.3),
            nn.PReLU(4 * ndf),
            nn.utils.spectral_norm(nn.Linear(ndf * 4, 1)),
            LearnableSigmoid(1),
        )

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        return self.layers(xy)


#------------------------------------------------------------------GENERATOR---------------------------------------------------------
@SerializableModule.register_model('CMGAN generator')
class CMGANGenerator(SerializableModule):
    def __init__(self, num_channel=64, num_features=201):
        super(CMGANGenerator, self).__init__()
        self.dense_encoder = DenseEncoder(in_channel=3, channels=num_channel)

        self.TSCB_1 = TSCB(num_channel=num_channel)
        self.TSCB_2 = TSCB(num_channel=num_channel)
        self.TSCB_3 = TSCB(num_channel=num_channel)
        self.TSCB_4 = TSCB(num_channel=num_channel)

        self.mask_decoder = MaskDecoder(
            num_features, num_channel=num_channel, out_channel=1
        )
        self.complex_decoder = ComplexDecoder(num_channel=num_channel)

    def forward(self, x):
        mag = torch.sqrt(x[:, 0, :, :] ** 2 + x[:, 1, :, :] ** 2).unsqueeze(1)
        noisy_phase = torch.angle(
            torch.complex(x[:, 0, :, :], x[:, 1, :, :])
        ).unsqueeze(1)
        x_in = torch.cat([mag, x], dim=1)

        out_1 = self.dense_encoder(x_in)
        out_2 = self.TSCB_1(out_1)
        out_3 = self.TSCB_2(out_2)
        out_4 = self.TSCB_3(out_3)
        out_5 = self.TSCB_4(out_4)

        mask = self.mask_decoder(out_5)
        out_mag = mask * mag

        complex_out = self.complex_decoder(out_5)
        mag_real = out_mag * torch.cos(noisy_phase)
        mag_imag = out_mag * torch.sin(noisy_phase)
        final_real = mag_real + complex_out[:, 0, :, :].unsqueeze(1)
        final_imag = mag_imag + complex_out[:, 1, :, :].unsqueeze(1)

        return final_real, final_imag
    
    

################################################################# TPTGAN ##########################################################
'''
- Generator
    - Encoder
        - Conv + Norm + Prelu
        - DenseBlockTPT
        - Norm + PreLU
    - Transformers latent space
        - Dual transformer
            - Uses transformer encoder with multihead + GRU + Bidirectional
    - Decoders
        - Magnitude (not used??????????????)
            - Dense
            - SCPConvTranspose + Norm + Prelu
        - Complex decoder(?)
            - Dense 
            - Norm + PreLU + Conv

    - Does not use classes for encoder, latent space and decoders
        - instead uses some modules for encoder (including denseblocktpt)
        - dualtransformer wchich uses transformer encoder for the latent space
        - some modules for decoders but magnitude decoder is not used


- Discriminator
    - Conv + Norm + PreLU  x 4 --> prácticamente igual que el de CMGAN

    No other class
'''

#---------------------------------------------------------------DISCRIMINATOR -------------------------------------------------------
@SerializableModule.register_model('TPTGAN discriminator')
class TPTGANDiscriminator(SerializableModule):
    def __init__(self, ndf, in_channel=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channel, ndf, (3,3), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(ndf, affine=True),
            nn.PReLU(ndf),
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, (3,3), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.PReLU(2 * ndf),
            nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, (3,3), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.PReLU(4 * ndf),
            nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, (3,3), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.PReLU(8 * ndf),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(ndf * 8, ndf * 4)),
            nn.Dropout(0.3),
            nn.PReLU(4 * ndf),
            nn.utils.spectral_norm(nn.Linear(ndf * 4, 1)),
            LearnableSigmoid(1)
        )

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        return self.layers(xy)

#----------------------------------------------------------------GENERATOR----------------------------------------------------------
@SerializableModule.register_model('TPTGAN generator')
class TPTGANGenerator(SerializableModule):
    def __init__(self, num_channel=64, num_features=201):
        super(TPTGANGenerator, self).__init__()
        self.dense_encoder = DenseEncoder(in_channel=3, channels=num_channel)

        self.myDPT_1 = DPTBlock(num_channel, 4)
        self.myDPT_2 = DPTBlock(num_channel, 4)
        self.myDPT_3 = DPTBlock(num_channel, 4)
        self.myDPT_4 = DPTBlock(num_channel, 4)

        self.mask_decoder = MaskDecoder(
            num_features, num_channel=num_channel, out_channel=1
        )
        self.complex_decoder = ComplexDecoder(num_channel=num_channel)

    def forward(self, x):
        mag = torch.sqrt(x[:, 0, :, :] ** 2 + x[:, 1, :, :] ** 2).unsqueeze(1)
        noisy_phase = torch.angle(
            torch.complex(x[:, 0, :, :], x[:, 1, :, :])
        ).unsqueeze(1)
        x_in = torch.cat([mag, x], dim=1)

        out_1 = self.dense_encoder(x_in)
        out_2 = self.myDPT_1(out_1)
        out_3 = self.myDPT_2(out_2)
        out_4 = self.myDPT_3(out_3)
        out_5 = self.myDPT_4(out_4)

        mask = self.mask_decoder(out_5)
        out_mag = mask * mag

        complex_out = self.complex_decoder(out_5)
        mag_real = out_mag * torch.cos(noisy_phase)
        mag_imag = out_mag * torch.sin(noisy_phase)
        final_real = mag_real + complex_out[:, 0, :, :].unsqueeze(1)
        final_imag = mag_imag + complex_out[:, 1, :, :].unsqueeze(1)

        return final_real, final_imag


#----------------------------------------------------------------DISCRIMINATOR--------------------------------------------------------
@SerializableModule.register_model('MP-SENet discriminator')
class MPSENETDiscriminator(SerializableModule):
    def __init__(self, dim=16, in_channel=2):
        super(MPSENETDiscriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channel, dim, (3,3), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(dim, affine=True),
            nn.PReLU(dim),
            nn.utils.spectral_norm(nn.Conv2d(dim, dim*2, (3,3), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(dim*2, affine=True),
            nn.PReLU(dim*2),
            nn.utils.spectral_norm(nn.Conv2d(dim*2, dim*4, (3,3), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(dim*4, affine=True),
            nn.PReLU(dim*4),
            nn.utils.spectral_norm(nn.Conv2d(dim*4, dim*8, (3,3), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(dim*8, affine=True),
            nn.PReLU(dim*8),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(dim*8, dim*4)),
            nn.Dropout(0.3),
            nn.PReLU(dim*4),
            nn.utils.spectral_norm(nn.Linear(dim*4, 1)),
            LearnableSigmoid_1d(1)
        )

    def forward(self, x, y):
        xy = torch.stack((x, y), dim=1)
        return self.layers(xy)
    

#----------------------------------------------------------------- GENERATOR------------------------------------------------------------
@SerializableModule.register_model('MP-SENet generator')
class MPSENETGenerator(SerializableModule):
    def __init__(self, num_tscblocks=4):
        super(MPSENETGenerator, self).__init__()
        self.num_tscblocks = num_tscblocks
        self.dense_encoder = DenseEncoderdense(in_channel=2)

        self.TSConformer = nn.ModuleList([])
        for i in range(num_tscblocks):
            self.TSConformer.append(TSConformerBlock())
        
        self.mask_decoder = MaskDecoderdense(out_channel=1)
        self.phase_decoder = PhaseDecoder(out_channel=1)

    def forward(self, noisy_mag, noisy_pha): # [B, F, T]
        noisy_pha = noisy_pha.unsqueeze(-1).permute(0, 3, 2, 1)
        noisy_mag = noisy_mag.permute(0, 1, 3, 2)
        x = torch.cat((noisy_mag, noisy_pha), dim=1) # [B, 2, T, F]
        x = self.dense_encoder(x)
        for i in range(self.num_tscblocks):
            x = self.TSConformer[i](x)
        denoised_mag = (noisy_mag * self.mask_decoder(x)).permute(0, 3, 2, 1).squeeze(-1)
        denoised_pha = self.phase_decoder(x).permute(0, 3, 2, 1).squeeze(-1)
        denoised_com = torch.stack((denoised_mag*torch.cos(denoised_pha),
                                    denoised_mag*torch.sin(denoised_pha)), dim=-1)

        return denoised_mag, denoised_pha, denoised_com
    

#############################################LitGAN
class EncoderLit(nn.Module):
    def __init__(self, ks=3, depth=4, in_channel=2, dense_channel=4):
        super(EncoderLit, self).__init__()
        self.depth=depth
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(2, 4, (1, 1)),
            nn.InstanceNorm2d(4, affine=True),
            nn.PReLU(4))
        
        kernel_size=(3, 3)
        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size, dilation=(1, 1),
                        padding=get_padding_2d(kernel_size, (1, 1))),
            nn.InstanceNorm2d(8, affine=True),
            nn.PReLU(8)
        )
        #kernel_size=(6, 6) #si lo cambias cambia la ultima dimension 
        self.dense_conv_3 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size, dilation=(2, 1),
                        padding=get_padding_2d(kernel_size, (2, 1))),
            nn.InstanceNorm2d(16, affine=True),
            nn.PReLU(16)
        )
        #kernel_size=(9, 9)
        self.dense_conv_4 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size, dilation=(4, 1),
                        padding=get_padding_2d(kernel_size, (4, 1))),

            nn.InstanceNorm2d(32, affine=True),
            nn.PReLU(32)
        )
        #kernel_size=(12, 12)
        self.dense_conv_5 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size, dilation=(8, 1),
                        padding=get_padding_2d(kernel_size, (8, 1))),
            nn.InstanceNorm2d(64, affine=True),
            nn.PReLU(64)
        )
        #careful uses dense block instead of dilateddense


    def forward(self, x):
        x = self.dense_conv_1(x) 
        x = self.dense_conv_2(x) 
        x = self.dense_conv_3(x) 
        x = self.dense_conv_4(x) 
        x = self.dense_conv_5(x) 
        return x


####################
class TSLitConformerBlock(nn.Module):
    def __init__(self,dense_channel=64):
        super(TSLitConformerBlock, self).__init__()
        self.time_conformer = LitConformer(dim=dense_channel,  n_head=4, ccm_kernel_size=31, 
                                             ffm_dropout=0.2, attn_dropout=0.2)
        self.freq_conformer = LitConformer(dim=dense_channel,  n_head=4, ccm_kernel_size=31, 
                                             ffm_dropout=0.2, attn_dropout=0.2)

    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        x = self.time_conformer(x) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        x = self.freq_conformer(x) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x


class LitConformer(nn.Module):
    def __init__(self, dim, n_head=8, ffm_mult=4, ccm_expansion_factor=2, ccm_kernel_size=31, 
                 ffm_dropout=0., attn_dropout=0., ccm_dropout=0.):
        super(LitConformer, self).__init__()
        self.ffm1 = FeedForward(dim, ffm_mult, dropout=ffm_dropout)
        self.attn = AttentionModule(dim, n_head, dropout=attn_dropout) 
        self.ccm = ConvolutionModule(dim, ccm_expansion_factor, ccm_kernel_size, dropout=ccm_dropout)
        self.ffm2 = FeedForward(dim, ffm_mult, dropout=ffm_dropout)
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + 0.5 * self.ffm1(x)
        x = x + self.attn(x)
        x = x + self.ccm(x)
        x = x + 0.5 * self.ffm2(x)
        x = self.post_norm(x)
        return x

#############
class MaskDecoderLit(nn.Module):
    def __init__(self, ks=15, depth=4, in_channels=128, dense_channel=64):
        super(MaskDecoderLit, self).__init__()
        kernel_size=(3, 3)
        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size, dilation=(8, 1),
                        padding=get_padding_2d(kernel_size, (8, 1))),
            nn.InstanceNorm2d(32, affine=True),
            nn.PReLU(32)
        )
        #kernel_size=(6, 6) #si lo cambias cambia la ultima dimension 
        self.dense_conv_3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size, dilation=(4, 1),
                        padding=get_padding_2d(kernel_size, (4, 1))),
            nn.InstanceNorm2d(16, affine=True),
            nn.PReLU(16)
        )
        #kernel_size=(9, 9)
        self.dense_conv_4 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size, dilation=(2, 1),
                        padding=get_padding_2d(kernel_size, (2, 1))),

            nn.InstanceNorm2d(8, affine=True),
            nn.PReLU(8)
        )
        #kernel_size=(12, 12)
        self.dense_conv_5 = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size, dilation=(1, 1),
                        padding=get_padding_2d(kernel_size, (1, 1))),
            nn.InstanceNorm2d(4, affine=True),
            nn.PReLU(4)
        )

        self.mask_conv = nn.Sequential(
            #nn.ConvTranspose2d(4, 4, (1, 3), (1, 2)),
            nn.Conv2d(4, 1, (1, 1)),

            #nn.InstanceNorm2d(1, affine=True),
            #nn.PReLU(1),
            #nn.Conv2d(1, 1, (1, 1))
        )
        self.lsigmoid = LearnableSigmoid_2d(512//2+1) #n_fft=400

    def forward(self, x):
        x = self.dense_conv_2(x)
        x = self.dense_conv_3(x)
        x = self.dense_conv_4(x)
        x = self.dense_conv_5(x)
        x = self.mask_conv(x)
        x = x.permute(0, 3, 2, 1).squeeze(-1)
        x = self.lsigmoid(x).permute(0, 2, 1).unsqueeze(1)
        return x


class PhaseDecoderLit(nn.Module):
    def __init__(self, ks=15, depth=4, in_channels=128, dense_channel=64):
        super(PhaseDecoderLit, self).__init__()
        self.depth=depth
        kernel_size=(3,3)
        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size, dilation=(8, 1),
                        padding=get_padding_2d(kernel_size, (8, 1))),
            nn.InstanceNorm2d(32, affine=True),
            nn.PReLU(32)
        )
        #kernel_size=(6, 6) #si lo cambias cambia la ultima dimension 
        self.dense_conv_3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size, dilation=(4, 1),
                        padding=get_padding_2d(kernel_size, (4, 1))),
            nn.InstanceNorm2d(16, affine=True),
            nn.PReLU(16)
        )
        #kernel_size=(9, 9)
        self.dense_conv_4 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size, dilation=(2, 1),
                        padding=get_padding_2d(kernel_size, (2, 1))),

            nn.InstanceNorm2d(8, affine=True),
            nn.PReLU(8)
        )
        #kernel_size=(12, 12)
        self.dense_conv_5 = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size, dilation=(1, 1),
                        padding=get_padding_2d(kernel_size, (1, 1))),
            nn.InstanceNorm2d(4, affine=True),
            nn.PReLU(4)
        )
        

        self.phase_conv_r = nn.Conv2d(4, 1, (1, 1))
        self.phase_conv_i = nn.Conv2d(4, 1, (1, 1))

    def forward(self, x):
        x = self.dense_conv_2(x)
        x = self.dense_conv_3(x)
        x = self.dense_conv_4(x)
        x = self.dense_conv_5(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        return x
#############################################3
class MaskDecoderMid(nn.Module):
    def __init__(self, ks=15, depth=4, in_channels=128, dense_channel=64):
        super(MaskDecoderMid, self).__init__()
        kernel_size=(3, 3)
        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size, dilation=(8, 1),
                        padding=get_padding_2d(kernel_size, (8, 1))),
            nn.InstanceNorm2d(32, affine=True),
            nn.PReLU(32)
        )
        #kernel_size=(6, 6) #si lo cambias cambia la ultima dimension 
        self.dense_conv_3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size, dilation=(4, 1),
                        padding=get_padding_2d(kernel_size, (4, 1))),
            nn.InstanceNorm2d(16, affine=True),
            nn.PReLU(16)
        )
        #kernel_size=(9, 9)
        self.dense_conv_4 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size, dilation=(2, 1),
                        padding=get_padding_2d(kernel_size, (2, 1))),

            nn.InstanceNorm2d(8, affine=True),
            nn.PReLU(8)
        )
        #kernel_size=(12, 12)
        self.dense_conv_5 = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size, dilation=(1, 1),
                        padding=get_padding_2d(kernel_size, (1, 1))),
            nn.InstanceNorm2d(4, affine=True),
            nn.PReLU(4)
        )

        self.mask_conv = nn.Sequential(
            nn.Conv2d(4, 1, (1, 1))
        )
        self.lsigmoid = LearnableSigmoid_2d(512//2+1) #n_fft=400

    def forward(self, x):
        x = self.dense_conv_2(x)
        x = self.dense_conv_3(x)
        x = self.dense_conv_4(x)
        x = self.dense_conv_5(x)
        x = self.mask_conv(x)
        x = x.permute(0, 3, 2, 1).squeeze(-1)
        x = self.lsigmoid(x).permute(0, 2, 1).unsqueeze(1)
        return x


class PhaseDecoderMid(nn.Module):
    def __init__(self, ks=15, depth=4, in_channels=128, dense_channel=64):
        super(PhaseDecoderMid, self).__init__()
        self.depth=depth
        kernel_size=(3,3)
        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size, dilation=(8, 1),
                        padding=get_padding_2d(kernel_size, (8, 1))),
            nn.InstanceNorm2d(32, affine=True),
            nn.PReLU(32)
        )
        #kernel_size=(6, 6) #si lo cambias cambia la ultima dimension 
        self.dense_conv_3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size, dilation=(4, 1),
                        padding=get_padding_2d(kernel_size, (4, 1))),
            nn.InstanceNorm2d(16, affine=True),
            nn.PReLU(16)
        )
        #kernel_size=(9, 9)
        self.dense_conv_4 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size, dilation=(2, 1),
                        padding=get_padding_2d(kernel_size, (2, 1))),

            nn.InstanceNorm2d(8, affine=True),
            nn.PReLU(8)
        )
        #kernel_size=(12, 12)
        self.dense_conv_5 = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size, dilation=(1, 1),
                        padding=get_padding_2d(kernel_size, (1, 1))),
            nn.InstanceNorm2d(4, affine=True),
            nn.PReLU(4)
        )
        
        self.phase_conv_r = nn.Conv2d(4, 1, (1, 1))
        self.phase_conv_i = nn.Conv2d(4, 1, (1, 1))

    def forward(self, x):
        x = self.dense_conv_2(x)
        x = self.dense_conv_3(x)
        x = self.dense_conv_4(x)
        x = self.dense_conv_5(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        return x

#########################################################Big

class MaskDecoderBig(nn.Module):
    def __init__(self, ks=15, depth=4, in_channels=128, dense_channel=64):
        super(MaskDecoderBig, self).__init__()
        kernel_size=(3, 3)
        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size, dilation=(8, 1),
                        padding=get_padding_2d(kernel_size, (8, 1))),
            nn.InstanceNorm2d(128, affine=True),
            nn.PReLU(128)
        )
        #kernel_size=(6, 6) #si lo cambias cambia la ultima dimension 
        self.dense_conv_3 = nn.Sequential(
            nn.Conv2d(128, 96, kernel_size, dilation=(4, 1),
                        padding=get_padding_2d(kernel_size, (4, 1))),
            nn.InstanceNorm2d(96, affine=True),
            nn.PReLU(96)
        )
        #kernel_size=(9, 9)
        self.dense_conv_4 = nn.Sequential(
            nn.Conv2d(96, 48, kernel_size, dilation=(2, 1),
                        padding=get_padding_2d(kernel_size, (2, 1))),

            nn.InstanceNorm2d(48, affine=True),
            nn.PReLU(48)
        )
        #kernel_size=(12, 12)
        self.dense_conv_5 = nn.Sequential(
            nn.Conv2d(48, 16, kernel_size, dilation=(1, 1),
                        padding=get_padding_2d(kernel_size, (1, 1))),
            nn.InstanceNorm2d(16, affine=True),
            nn.PReLU(16)
        )

        self.mask_conv = nn.Sequential(
            #nn.ConvTranspose2d(4, 4, (1, 3), (1, 2)),
            nn.Conv2d(16, 1, (1, 1)),

            #nn.InstanceNorm2d(1, affine=True),
            #nn.PReLU(1),
            #nn.Conv2d(1, 1, (1, 1))
        )
        self.lsigmoid = LearnableSigmoid_2d(512//2+1) #n_fft=400

    def forward(self, x):
        x = self.dense_conv_2(x)
        x = self.dense_conv_3(x)
        x = self.dense_conv_4(x)
        x = self.dense_conv_5(x)
        x = self.mask_conv(x)
        x = x.permute(0, 3, 2, 1).squeeze(-1)
        x = self.lsigmoid(x).permute(0, 2, 1).unsqueeze(1)
        return x


class PhaseDecoderBig(nn.Module):
    def __init__(self, ks=15, depth=4, in_channels=128, dense_channel=64):
        super(PhaseDecoderBig, self).__init__()
        self.depth=depth
        kernel_size=(3,3)
        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size, dilation=(8, 1),
                        padding=get_padding_2d(kernel_size, (8, 1))),
            nn.InstanceNorm2d(128, affine=True),
            nn.PReLU(128)
        )
        #kernel_size=(6, 6) #si lo cambias cambia la ultima dimension 
        self.dense_conv_3 = nn.Sequential(
            nn.Conv2d(128, 96, kernel_size, dilation=(4, 1),
                        padding=get_padding_2d(kernel_size, (4, 1))),
            nn.InstanceNorm2d(96, affine=True),
            nn.PReLU(96)
        )
        #kernel_size=(9, 9)
        self.dense_conv_4 = nn.Sequential(
            nn.Conv2d(96, 48, kernel_size, dilation=(2, 1),
                        padding=get_padding_2d(kernel_size, (2, 1))),

            nn.InstanceNorm2d(48, affine=True),
            nn.PReLU(48)
        )
        #kernel_size=(12, 12)
        self.dense_conv_5 = nn.Sequential(
            nn.Conv2d(48, 16, kernel_size, dilation=(1, 1),
                        padding=get_padding_2d(kernel_size, (1, 1))),
            nn.InstanceNorm2d(16, affine=True),
            nn.PReLU(16)
        )
        

        self.phase_conv_r = nn.Conv2d(16, 1, (1, 1))
        self.phase_conv_i = nn.Conv2d(16, 1, (1, 1))

    def forward(self, x):
        x = self.dense_conv_2(x)
        x = self.dense_conv_3(x)
        x = self.dense_conv_4(x)
        x = self.dense_conv_5(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        return x


class EncoderBig(nn.Module):
    def __init__(self, ks=3, depth=4, in_channel=2, dense_channel=4):
        super(EncoderBig, self).__init__()
        self.depth=depth
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(2, 16, (1, 1)),
            nn.InstanceNorm2d(16, affine=True),
            nn.PReLU(16))
        
        kernel_size=(3, 3)
        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(16, 48, kernel_size, dilation=(1, 1),
                        padding=get_padding_2d(kernel_size, (1, 1))),
            nn.InstanceNorm2d(48, affine=True),
            nn.PReLU(48)
        )
        #kernel_size=(6, 6) #si lo cambias cambia la ultima dimension 
        self.dense_conv_3 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size, dilation=(2, 1),
                        padding=get_padding_2d(kernel_size, (2, 1))),
            nn.InstanceNorm2d(96, affine=True),
            nn.PReLU(96)
        )
        #kernel_size=(9, 9)
        self.dense_conv_4 = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size, dilation=(4, 1),
                        padding=get_padding_2d(kernel_size, (4, 1))),

            nn.InstanceNorm2d(128, affine=True),
            nn.PReLU(128)
        )
        #kernel_size=(12, 12)
        self.dense_conv_5 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size, dilation=(8, 1),
                        padding=get_padding_2d(kernel_size, (8, 1))),
            nn.InstanceNorm2d(64, affine=True),
            nn.PReLU(64)
        )
        #careful uses dense block instead of dilateddense


    def forward(self, x):
        x = self.dense_conv_1(x) 
        x = self.dense_conv_2(x) 
        x = self.dense_conv_3(x) 
        x = self.dense_conv_4(x) 
        x = self.dense_conv_5(x) 
        return x


#----------------------------------------------------------------- GENERATOR------------------------------------------------------------
@SerializableModule.register_model('LitGAN generator')
class LitGANGenerator(SerializableModule):
    def __init__(self):
        super(LitGANGenerator, self).__init__()
        self.dense_encoder = EncoderLit()
        self.conformer = TSLitConformerBlock()
        self.mask_decoder = MaskDecoderLit()
        self.phase_decoder = PhaseDecoderLit()

    def forward(self, noisy_mag, noisy_pha): # [B, F, T]
        noisy_pha = noisy_pha.unsqueeze(-1).permute(0, 3, 2, 1)
        noisy_mag = noisy_mag.permute(0, 1, 3, 2)
        x = torch.cat((noisy_mag, noisy_pha), dim=1) # [B, 2, T, F]
        x = self.dense_encoder(x)

        x = self.conformer(x)

        denoised_mag = (noisy_mag * self.mask_decoder(x)).permute(0, 3, 2, 1).squeeze(-1)
        denoised_pha = self.phase_decoder(x).permute(0, 3, 2, 1).squeeze(-1)

        return denoised_mag, denoised_pha

#----------------------------------------------------------------DISCRIMINATOR--------------------------------------------------------
@SerializableModule.register_model('LitGAN discriminator')
class LitGANDiscriminator(SerializableModule):
    def __init__(self, dim=16, in_channel=2):
        super(LitGANDiscriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channel, dim, (3,3), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(dim, affine=True),
            nn.PReLU(dim),
            nn.utils.spectral_norm(nn.Conv2d(dim, dim*2, (3,3), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(dim*2, affine=True),
            nn.PReLU(dim*2),
            nn.utils.spectral_norm(nn.Conv2d(dim*2, dim*4, (3,3), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(dim*4, affine=True),
            nn.PReLU(dim*4),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(dim*4, dim*2)),
            nn.Dropout(0.3),
            nn.PReLU(dim*2),
            nn.utils.spectral_norm(nn.Linear(dim*2, 1)),
            LearnableSigmoid_1d(1)
        )

    def forward(self, x, y):
        xy = torch.stack((x, y), dim=1)
        return self.layers(xy)
    
#______________________________________________________________MIDGAN
@SerializableModule.register_model('MidGAN generator')
class MidGANGenerator(SerializableModule):
    def __init__(self):
        super(MidGANGenerator, self).__init__()
        self.dense_encoder = EncoderLit()
        self.conformer1 = TSLitConformerBlock()
        self.conformer2 = TSLitConformerBlock()
        self.mask_decoder = MaskDecoderMid()
        self.phase_decoder = PhaseDecoderMid()

    def forward(self, noisy_mag, noisy_pha): # [B, F, T]
        noisy_pha = noisy_pha.unsqueeze(-1).permute(0, 3, 2, 1)
        noisy_mag = noisy_mag.permute(0, 1, 3, 2)
        x = torch.cat((noisy_mag, noisy_pha), dim=1) # [B, 2, T, F]
        x = self.dense_encoder(x)

        x = self.conformer1(x)
        x = self.conformer2(x)

        denoised_mag = (noisy_mag * self.mask_decoder(x)).permute(0, 3, 2, 1).squeeze(-1)
        denoised_pha = self.phase_decoder(x).permute(0, 3, 2, 1).squeeze(-1)

        return denoised_mag, denoised_pha

#----------------------------------------------------------------DISCRIMINATOR--------------------------------------------------------
@SerializableModule.register_model('MidGAN discriminator')
class MidGANDiscriminator(SerializableModule):
    def __init__(self, dim=16, in_channel=2):
        super(MidGANDiscriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channel, dim, (3,3), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(dim, affine=True),
            nn.PReLU(dim),
            nn.utils.spectral_norm(nn.Conv2d(dim, dim*2, (3,3), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(dim*2, affine=True),
            nn.PReLU(dim*2),
            nn.utils.spectral_norm(nn.Conv2d(dim*2, dim*4, (3,3), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(dim*4, affine=True),
            nn.PReLU(dim*4),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(dim*4, dim*2)),
            nn.Dropout(0.3),
            nn.PReLU(dim*2),
            nn.utils.spectral_norm(nn.Linear(dim*2, 1)),
            LearnableSigmoid_1d(1)
        )

    def forward(self, x, y):
        xy = torch.stack((x, y), dim=1)
        return self.layers(xy)


######################################### BIGGAN
@SerializableModule.register_model('GigGAN generator')
class GigGANGenerator(SerializableModule):
    def __init__(self):
        super(BigGANGenerator, self).__init__()
        self.dense_encoder = EncoderBig()
        self.conformer = TSLitConformerBlock(dense_channel=64)
        self.mask_decoder = MaskDecoderBig()
        self.phase_decoder = PhaseDecoderBig()

    def forward(self, noisy_mag, noisy_pha): # [B, F, T]
        noisy_pha = noisy_pha.unsqueeze(-1).permute(0, 3, 2, 1)
        noisy_mag = noisy_mag.permute(0, 1, 3, 2)
        x = torch.cat((noisy_mag, noisy_pha), dim=1) # [B, 2, T, F]
        x = self.dense_encoder(x)

        x = self.conformer(x)

        denoised_mag = (noisy_mag * self.mask_decoder(x)).permute(0, 3, 2, 1).squeeze(-1)
        denoised_pha = self.phase_decoder(x).permute(0, 3, 2, 1).squeeze(-1)

        return denoised_mag, denoised_pha

#----------------------------------------------------------------DISCRIMINATOR--------------------------------------------------------
@SerializableModule.register_model('GigGAN discriminator')
class GigGANDiscriminator(SerializableModule):
    def __init__(self, dim=16, in_channel=2):
        super(BigGANDiscriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channel, dim, (3,3), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(dim, affine=True),
            nn.PReLU(dim),
            nn.utils.spectral_norm(nn.Conv2d(dim, dim*2, (3,3), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(dim*2, affine=True),
            nn.PReLU(dim*2),
            nn.utils.spectral_norm(nn.Conv2d(dim*2, dim*4, (3,3), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(dim*4, affine=True),
            nn.PReLU(dim*4),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(dim*4, dim*2)),
            nn.Dropout(0.3),
            nn.PReLU(dim*2),
            nn.utils.spectral_norm(nn.Linear(dim*2, 1)),
            LearnableSigmoid_1d(1)
        )

    def forward(self, x, y):
        xy = torch.stack((x, y), dim=1)
        return self.layers(xy)

######################################### BIGGAN
@SerializableModule.register_model('BigGAN generator')
class BigGANGenerator(SerializableModule):
    def __init__(self):
        super(BigGANGenerator, self).__init__()
        self.dense_encoder = DenseEncoderdense(in_channel=2, dense_channel=32)
        self.conformer= TSConformerBlock(dense_channel=32)
        self.mask_decoder = MaskDecoderdense(out_channel=1, dense_channel=32)
        self.phase_decoder = PhaseDecoder(out_channel=1, dense_channel=32)


    def forward(self, noisy_mag, noisy_pha): # [B, F, T]
        noisy_pha = noisy_pha.unsqueeze(-1).permute(0, 3, 2, 1)
        noisy_mag = noisy_mag.permute(0, 1, 3, 2)
        x = torch.cat((noisy_mag, noisy_pha), dim=1) # [B, 2, T, F]
        x = self.dense_encoder(x)

        x = self.conformer(x)

        denoised_mag = (noisy_mag * self.mask_decoder(x)).permute(0, 3, 2, 1).squeeze(-1)
        denoised_pha = self.phase_decoder(x).permute(0, 3, 2, 1).squeeze(-1)

        return denoised_mag, denoised_pha

#----------------------------------------------------------------DISCRIMINATOR--------------------------------------------------------
@SerializableModule.register_model('BigGAN discriminator')
class BigGANDiscriminator(SerializableModule):
    def __init__(self, dim=16, in_channel=2):
        super(BigGANDiscriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channel, dim, (3,3), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(dim, affine=True),
            nn.PReLU(dim),
            nn.utils.spectral_norm(nn.Conv2d(dim, dim*2, (3,3), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(dim*2, affine=True),
            nn.PReLU(dim*2),
            nn.utils.spectral_norm(nn.Conv2d(dim*2, dim*4, (3,3), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(dim*4, affine=True),
            nn.PReLU(dim*4),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(dim*4, dim*2)),
            nn.Dropout(0.3),
            nn.PReLU(dim*2),
            nn.utils.spectral_norm(nn.Linear(dim*2, 1)),
            LearnableSigmoid_1d(1)
        )

    def forward(self, x, y):
        xy = torch.stack((x, y), dim=1)
        return self.layers(xy)
    
########################## MINIGAN
@SerializableModule.register_model('MiniGAN generator')
class MiniGANGenerator(SerializableModule):
    def __init__(self):
        super(MiniGANGenerator, self).__init__()
        self.dense_encoder = DenseEncoderdense(in_channel=2, dense_channel=32)
        self.conformer= TSConformerBlock(dense_channel=32)
        self.mask_decoder = MaskDecoderdense(out_channel=1, dense_channel=32)
        self.phase_decoder = PhaseDecoder(out_channel=1, dense_channel=32)


    def forward(self, noisy_mag, noisy_pha): # [B, F, T]
        noisy_pha = noisy_pha.unsqueeze(-1).permute(0, 3, 2, 1)
        noisy_mag = noisy_mag.permute(0, 1, 3, 2)
        x = torch.cat((noisy_mag, noisy_pha), dim=1) # [B, 2, T, F]
        x = self.dense_encoder(x)

        x = self.conformer(x)

        denoised_mag = (noisy_mag * self.mask_decoder(x)).permute(0, 3, 2, 1).squeeze(-1)
        denoised_pha = self.phase_decoder(x).permute(0, 3, 2, 1).squeeze(-1)

        return denoised_mag, denoised_pha

#----------------------------------------------------------------DISCRIMINATOR--------------------------------------------------------
@SerializableModule.register_model('MiniGAN discriminator')
class MiniGANDiscriminator(SerializableModule):
    def __init__(self, dim=16, in_channel=2):
        super(MiniGANDiscriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channel, dim, (3,3), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(dim, affine=True),
            nn.PReLU(dim),
            nn.utils.spectral_norm(nn.Conv2d(dim, dim*2, (3,3), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(dim*2, affine=True),
            nn.PReLU(dim*2),
            nn.utils.spectral_norm(nn.Conv2d(dim*2, dim*4, (3,3), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(dim*4, affine=True),
            nn.PReLU(dim*4),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(dim*4, dim*2)),
            nn.Dropout(0.3),
            nn.PReLU(dim*2),
            nn.utils.spectral_norm(nn.Linear(dim*2, 1)),
            LearnableSigmoid_1d(1)
        )

    def forward(self, x, y):
        xy = torch.stack((x, y), dim=1)
        return self.layers(xy)
    
#-------------------------------------------------- ABLATION

class TSConformerBlockother(nn.Module):
    def __init__(self, dense_channel=64):
        super(TSConformerBlockother, self).__init__()
        self.time_conformer = ConformerBlockModule(dim=dense_channel,  n_head=4, ccm_kernel_size=31, 
                                             ffm_dropout=0.2, attn_dropout=0.2)
        self.freq_conformer = ConformerBlockModule(dim=dense_channel,  n_head=4, ccm_kernel_size=31, 
                                             ffm_dropout=0.2, attn_dropout=0.2)

    def forward(self, x):
        b, c, t, f = x.size()

        x = x.permute(0, 2, 3, 1).contiguous().view(b*t, f, c)  # Shape: (b*t, f, c)
        x = self.freq_conformer(x) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2).contiguous()  # Shape: (b, c, t, f)

        x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)  # Shape: (b*f, t, c)
        x = self.time_conformer(x) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous()  # Shape: (b, t, f, c)
        x = x.permute(0, 3, 1, 2).contiguous()  # Shape: (b, c, t, f)

        return x


@SerializableModule.register_model('MiniGANmod1 generator')
class MiniGANmod1Generator(SerializableModule):
    def __init__(self):
        super(MiniGANmod1Generator, self).__init__()
        self.dense_encoder = DenseEncoderdense(in_channel=2, dense_channel=32)
        self.conformer= TSConformerBlockother(dense_channel=32)
        self.mask_decoder = MaskDecoderdense(out_channel=1, dense_channel=32)
        self.phase_decoder = PhaseDecoder(out_channel=1, dense_channel=32)


    def forward(self, noisy_mag, noisy_pha): # [B, F, T]
        noisy_pha = noisy_pha.unsqueeze(-1).permute(0, 3, 2, 1)
        noisy_mag = noisy_mag.permute(0, 1, 3, 2)
        x = torch.cat((noisy_mag, noisy_pha), dim=1) # [B, 2, T, F]
        x = self.dense_encoder(x)

        x = self.conformer(x)

        denoised_mag = (noisy_mag * self.mask_decoder(x)).permute(0, 3, 2, 1).squeeze(-1)
        denoised_pha = self.phase_decoder(x).permute(0, 3, 2, 1).squeeze(-1)

        return denoised_mag, denoised_pha

@SerializableModule.register_model('MiniGANmod2 generator')
class MiniGANmod2Generator(SerializableModule):
    def __init__(self):
        super(MiniGANmod2Generator, self).__init__()
        self.dense_encoder = DenseEncoderdense(in_channel=2, dense_channel=48)
        self.conformer= TSConformerBlock(dense_channel=48)
        self.mask_decoder = MaskDecoderdense(out_channel=1, dense_channel=48)
        self.phase_decoder = PhaseDecoder(out_channel=1, dense_channel=48)


    def forward(self, noisy_mag, noisy_pha): # [B, F, T]
        noisy_pha = noisy_pha.unsqueeze(-1).permute(0, 3, 2, 1)
        noisy_mag = noisy_mag.permute(0, 1, 3, 2)
        x = torch.cat((noisy_mag, noisy_pha), dim=1) # [B, 2, T, F]
        x = self.dense_encoder(x)

        x = self.conformer(x)

        denoised_mag = (noisy_mag * self.mask_decoder(x)).permute(0, 3, 2, 1).squeeze(-1)
        denoised_pha = self.phase_decoder(x).permute(0, 3, 2, 1).squeeze(-1)

        return denoised_mag, denoised_pha

