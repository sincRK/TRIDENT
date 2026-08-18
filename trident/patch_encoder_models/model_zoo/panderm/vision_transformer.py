"""
PanDerm Vision Transformer model.
Extracted from original implementation at https://github.com/SiyuanYan1/PanDerm
"""

import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import trunc_normal_

from .modeling_finetune import (
    LP_BatchNorm, Block, AttentiveBlock, PatchEmbed, RelativePositionBias
)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, init_values=0.1, use_rel_pos_bias=False,
                 use_shared_rel_pos_bias=False, use_mean_pooling=False,
                 init_scale=0.001, lin_probe=True, linear_type='standard', args=None):
        super().__init__()
        self.num_classes = num_classes
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim
        self.use_mean_pooling = use_mean_pooling

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = self.build_2d_sincos_position_embedding(embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(
                window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer, init_values=init_values,
                window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None)
            for i in range(depth)])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)

        self.lin_probe = lin_probe
        self.linear_type = linear_type

        if lin_probe:
            if self.linear_type == 'standard':
                self.fc_norm = None
            elif self.linear_type == 'attentive':
                self.query_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
                self.attentive_blocks = nn.ModuleList([
                    AttentiveBlock(
                        dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias,
                        qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=0, norm_layer=norm_layer)
                    for i in range(1)])
                self.fc_norm = LP_BatchNorm(embed_dim, affine=False)
        else:
            if use_mean_pooling:
                self.fc_norm = norm_layer(embed_dim)
            else:
                self.fc_norm = None

        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.cls_token, std=.02)
        if hasattr(self.head, 'weight'):
            trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

        if hasattr(self.head, 'weight'):
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

    def build_2d_sincos_position_embedding(self, embed_dim=768, temperature=10000.):
        h, w = self.patch_embed.patch_shape
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w),
                             torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

        # Add a CLS token position (zeros)
        cls_pos_emb = torch.zeros(1, 1, embed_dim)
        pos_emb = torch.cat([cls_pos_emb, pos_emb], dim=1)

        pos_embed = nn.Parameter(pos_emb)
        pos_embed.requires_grad = False

        return pos_embed

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(torch.sqrt(torch.tensor(2.0 * layer_id, dtype=param.dtype)))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, is_train=True):
        B, _, H, W = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, :x.shape[1], :].type_as(x).to(x.device).clone().detach()

        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)

        if self.use_mean_pooling:
            x = x[:, 1:].mean(dim=1)  # global average pooling
        else:
            x = x[:, 0]

        if self.lin_probe and self.linear_type == 'attentive':
            B, N, C = x.shape[0], 1, x.shape[-1]
            query = self.query_token.expand(B, -1, -1)

            for attentive_blk in self.attentive_blocks:
                query = attentive_blk(
                    query, x.unsqueeze(1), self.pos_embed[:, :1, :],
                    self.pos_embed[:, 1:, :].mean(dim=1, keepdim=True))

            x = query.squeeze(1)

        outcome = self.norm(x)

        if self.fc_norm is not None:
            if hasattr(self.fc_norm, 'forward') and 'is_train' in self.fc_norm.forward.__code__.co_varnames:
                outcome = self.fc_norm(outcome, is_train)
            else:
                outcome = self.fc_norm(outcome)

        return outcome

    def forward(self, x):
        x = self.forward_features(x, is_train=False)
        x = self.head(x)
        return x


def panderm_large_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


def panderm_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model
