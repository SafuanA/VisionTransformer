from functools import partial
from typing import Optional
import torch
import torch.nn as nn
import numpy as np
#import timm.models.vision_transformer
#from timm.models.vision_transformer import Block
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils.features import Mel_Spectrogram
from utils.amsoftmax import amsoftmax
from utils.Helpers import getTimeDimension

def vit_s(pretrain, n_classes, shuffle_type, seconds, hop, cls_pos, FcLayer = False):
    timeDim = getTimeDimension(seconds, hop)
    return VisionTransformer(
                            img_size=(80, timeDim), patch_size=(80,1), in_chans=1,
                            embed_dim=192, depth=12, num_heads=3,
                            decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                            pretrain=pretrain, norm_file='files/mean_std_80.npy', n_classes=n_classes,
                            shuffle_type=shuffle_type, seconds=seconds, hop=hop, cls_pos=cls_pos, FcLayer = FcLayer)

def vit_m(pretrain, n_classes, shuffle_type, seconds, hop, cls_pos, FcLayer = False):
    timeDim = getTimeDimension(seconds, hop)
    return VisionTransformer(img_size=(80, timeDim), patch_size=(80,1), in_chans=1,
                            embed_dim=384, depth=12, num_heads=3,
                            decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                            pretrain=pretrain, norm_file='files/mean_std_80.npy', n_classes=n_classes,
                           shuffle_type=shuffle_type, seconds=seconds, hop=hop, cls_pos=cls_pos, FcLayer = FcLayer)

#use our own patchembeddings to release restrictions so we can make frame based prediction
class PatchEmbed(nn.Module):
    """ 2D Spectogram to Patch Embedding along the time axis"""
    def __init__(
            self,
            img_size = (80,104),
            patch_size = (80,1),
            in_chans: int = 1,
            embed_dim: int = 768,
            flatten: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.grid_size = (img_size[0]//patch_size[0],img_size[1]//patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
            
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.norm = nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        assert W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]})."

        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=(80, 104), patch_size=(80,1), in_chans=1,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 norm_file='files/mean_std.npy', pretrain=False, n_classes=1211, 
                 shuffle_type=None, seconds=2, FcLayer = False, hop=154, cls_pos=False, mask_type=1):
        super().__init__()
        self.pretrain = pretrain
        self.cls_pos = cls_pos
        # Mel Spectrogram
        self.mel = Mel_Spectrogram(shuffle_type=shuffle_type, seconds=seconds, mask_type=True, hop=hop)
        mean_std_file = np.load(norm_file, allow_pickle=True).item()
        AVAIL_GPUS = torch.cuda.device_count()
        if AVAIL_GPUS > 0:
            self.frame_mean = torch.Tensor(mean_std_file['frame_mean']).cuda()
            self.frame_std = torch.Tensor(mean_std_file['frame_std']).cuda()
        else:
            self.frame_mean = torch.Tensor(mean_std_file['frame_mean'])
            self.frame_std = torch.Tensor(mean_std_file['frame_std'])
        # PreTrainTransformer encoder specifics
        # -----------------
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        if pretrain:
            # PreTrainTransformer decoder specifics
            # ------------------
            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

            self.decoder_blocks = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(decoder_depth)])

            self.decoder_norm = norm_layer(decoder_embed_dim)
            self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size[0] * patch_size[1] * in_chans, bias=True) # decoder to patch
            
            #weight initialization
            self.initialize_weights()
        else:
            self.FcLayer = FcLayer
            if self.FcLayer:
                self.fc = nn.Linear(embed_dim, n_classes)
                self.fcNorm = nn.LayerNorm(n_classes) #aka batch norm
                self.pos_drop = nn.Dropout(p=0.1)
                self.loss_fun = amsoftmax(n_classes, n_classes)
            else:
                self.pos_drop = nn.Dropout(p=0.1)
                self.loss_fun = amsoftmax(embed_dim, n_classes)

    
    def mel_forward(self, x, train):
        x = self.mel(x, train)
        x = (x - self.frame_mean[None, :, None]) / self.frame_std[None, :, None]
        return x

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        if not self.cls_pos: #if true uses positional encoding for the cls stoken
            self.pos_embed.data.copy_(getPosEncoding(self.pos_embed.shape[-1], self.patch_embed.grid_size[1], True))
            self.decoder_pos_embed.data.copy_(getPosEncoding(self.decoder_pos_embed.shape[-1], self.patch_embed.grid_size[1], True))
        else:
            self.pos_embed.data.copy_(getPosEncoding(self.pos_embed.shape[-1], self.patch_embed.grid_size[1] + 1, False))
            self.decoder_pos_embed.data.copy_(getPosEncoding(self.decoder_pos_embed.shape[-1], self.patch_embed.grid_size[1] + 1, False))

            """pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.grid_size[0], self.patch_embed.grid_size[1], cls_token=True)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

            decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.patch_embed.grid_size[0], self.patch_embed.grid_size[1], cls_token=True)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))"""
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 1, H, W)
        x: (N, L, patch_size**2, 1)
        """
        assert imgs.shape[2] % self.patch_embed.grid_size[0] == 0 and imgs.shape[3] % self.patch_embed.grid_size[1] == 0

        h = self.patch_embed.grid_size[0]
        w = self.patch_embed.grid_size[1]
        p1 = self.patch_embed.patch_size[0]
        p2 = self.patch_embed.patch_size[1]
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p1, w, p2))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p1*p2, 1))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2, 1)
        imgs: (N, 1, H, W)
        """
        p1 = self.patch_embed.patch_size[0]
        p2 = self.patch_embed.patch_size[1]
        h = self.patch_embed.grid_size[0]
        w = self.patch_embed.grid_size[1]
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p1, p2, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p1, w * p2))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def forward_features(self, x):
        B = x.shape[0]

        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x[:, 0]
        return x

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 1, H, W]
        pred: [N, L, p*p]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        target = target[:,:,:,0]

        mse = (pred - target) ** 2
        mse = mse.mean(dim=-1)  # [N, L], mean loss per patch
        mse = (mse * mask).sum() / mask.sum()  # mean loss on removed patches

        #nce = torch.tensor(0.0).to(x.device)
        #correct = torch.tensor(0.0).to(x.device)
        #for i in np.arange(0, B):
        #    total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))  # e.g. size 100*100
        #    correct += torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, mask_patch, device=x.device)))  # correct is a tensor
        #    nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
        #acc = 1. * correct / (B * mask_patch)
        #nce = nce / (-1. * B * mask_patch)

        #loss = 10 * mse + nce

        return mse, target

    def forward(self, batch, train=False, test=False, mask_ratio=0.75):
        wav, label, spk_id_encoded, path = batch
        if torch.cuda.device_count() > 0:
            wav = wav.type(torch.Tensor).cuda()
        else:
            wav = wav.type(torch.Tensor)#.cuda() if not on cuda

        mel = self.mel_forward(wav, train)
        mel = mel.unsqueeze(1)
        if self.pretrain: 
            imgs = mel[:, :, :self.patch_embed.img_size[0], :self.patch_embed.img_size[1]]
            latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
            pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p]
            loss, tgt = self.forward_loss(imgs, pred, mask)
            return loss, pred, mask, tgt
        else:
            x = self.forward_features(mel)
            #simply using the cls token results in eer around 20-25 in our experiments,
            # attentive statisticl pooling might be give better results
            if self.FcLayer:
                x = self.fc(x)
                x = self.fcNorm(x)

            if test:
                return x, spk_id_encoded
            else:
                loss, acc, _ = self.loss_fun(x, spk_id_encoded)
                return (loss, acc), x, None, spk_id_encoded

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

def getPosEncoding(embed_dim, seq_len, cls_token=False):
    pe = torch.zeros(seq_len, embed_dim, requires_grad=False)
    positions = torch.arange(0, seq_len).unsqueeze(1).float()
    denominator = torch.exp(
        torch.arange(0, embed_dim, 2).float()
        * -(math.log(10000.0) / embed_dim)
    )

    pe[:, 0::2] = torch.sin(positions * denominator)
    pe[:, 1::2] = torch.cos(positions * denominator)
    if cls_token:
        pe = torch.concatenate([torch.zeros([1, embed_dim]), pe], axis=0)
    pe = pe.unsqueeze(0)
    return pe

def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w, cls_token=False):
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in:

'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929

`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270

`FlexiViT: One Model for All Patch Sizes`
    - https://arxiv.org/abs/2212.08013

The official jax code is released and available at
  * https://github.com/google-research/vision_transformer
  * https://github.com/google-research/big_vision

Acknowledgments:
  * The paper authors for releasing code and weights, thanks!
  * I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch
  * Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
  * Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020, Ross Wightman
"""
import logging
import math
from collections import OrderedDict
from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final
from timm.layers import Mlp, DropPath, use_fused_attn



class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x