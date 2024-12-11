import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.qkv = nn.Linear(dim, inner_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(inner_dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_ratio=4., dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, heads, dim_head, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_ratio=4., dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(TransformerBlock(dim, heads, dim_head, mlp_ratio, dropout))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Transformer(nn.Module):
    def __init__(
        self, 
        image_size=256,
        patch_size=16,
        in_channels=3,
        out_channels=1,
        embed_dim=512,
        depth=12,
        heads=8,
        dim_head=64,
        mlp_ratio=4.,
        dropout=0.1
    ):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim//2, kernel_size=7, stride=2, padding=3),
            nn.LayerNorm([embed_dim//2, image_size//2, image_size//2]),
            nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([embed_dim, image_size//4, image_size//4]),
            nn.GELU(),
        )

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        self.transformer_layers = nn.ModuleList([])
        for _ in range(depth):
            self.transformer_layers.append(
                nn.ModuleList([
                    TransformerBlock(embed_dim, heads, dim_head, mlp_ratio, dropout),
                    nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
                ])
            )

        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            Rearrange('b (h w) c -> b c h w', h=image_size//patch_size, w=image_size//patch_size),
            nn.ConvTranspose2d(embed_dim * 4, embed_dim * 2, kernel_size=2, stride=2),
            nn.LayerNorm([embed_dim * 2, image_size//patch_size * 2, image_size//patch_size * 2]),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim * 2, embed_dim, kernel_size=2, stride=2),
            nn.LayerNorm([embed_dim, image_size//patch_size * 4, image_size//patch_size * 4]),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim, out_channels, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.stem(x)
        
        # Patch embedding with spatial information
        B, C, H, W = x.shape
        x = self.to_patch_embedding(x)
        
        # Add CLS token and position embedding
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)
        
        # Multi-scale transformer processing
        for transformer, conv in self.transformer_layers:
            # Global attention
            x_global = transformer(x)
            
            # Local processing
            x_local = x[:, 1:, :].reshape(B, H, W, -1).permute(0, 3, 1, 2)
            x_local = conv(x_local)
            x_local = x_local.flatten(2).transpose(1, 2)
            
            # Combine global and local features
            x = torch.cat([x_global[:, :1, :], x_local], dim=1)

        x = x[:, 1:, :]
        x = self.decoder(x)
        return x

# class Transformer(nn.Module):
#     def __init__(
#         self, 
#         image_size=256,
#         patch_size=16,
#         in_channels=3,
#         out_channels=1,
#         embed_dim=512,
#         depth=12,
#         heads=8,
#         dim_head=64,
#         mlp_ratio=4.,
#         dropout=0.0
#     ):
#         super().__init__()
#         assert image_size % patch_size == 0, "Image size must be divisible by patch size"
#         num_patches = (image_size // patch_size) ** 2
#         patch_dim = in_channels * patch_size ** 2
        
#         # Patch embedding
#         self.to_patch_embedding = nn.Sequential(
#             Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
#             nn.Linear(patch_dim, embed_dim),
#             nn.LayerNorm(embed_dim)
#         )
        
#         # Position embedding
#         self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))
#         self.dropout = nn.Dropout(dropout)

#         # Transformer encoder
#         self.transformer = TransformerEncoder(
#             dim=embed_dim,
#             depth=depth,
#             heads=heads,
#             dim_head=dim_head,
#             mlp_ratio=mlp_ratio,
#             dropout=dropout
#         )

#         # Decoder (upsampling back to image size)
#         self.decoder = nn.Sequential(
#             nn.Linear(embed_dim, patch_dim),
#             Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', 
#                      h=image_size//patch_size, w=image_size//patch_size, 
#                      p1=patch_size, p2=patch_size),
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         # Patch embedding
#         x = self.to_patch_embedding(x)
        
#         # Add position embedding
#         x = x + self.pos_embedding
#         x = self.dropout(x)
        
#         # Transformer encoder
#         x = self.transformer(x)
        
#         # Decode to image
#         x = self.decoder(x)
        
#         return x

#     def init_weights(self, pretrained=None, strict=False):
#         if isinstance(pretrained, str):
#             state_dict = torch.load(pretrained, map_location='cpu')['state_dict']
#             self.load_state_dict(state_dict, strict=strict)
#         elif pretrained is None:
#             # Initialize weights
#             for m in self.modules():
#                 if isinstance(m, nn.Linear):
#                     nn.init.xavier_uniform_(m.weight)
#                     if m.bias is not None:
#                         nn.init.zeros_(m.bias)
#                 elif isinstance(m, nn.LayerNorm):
#                     nn.init.ones_(m.weight)
#                     nn.init.zeros_(m.bias)
#         else:
#             raise TypeError("'pretrained' must be a str or None.")