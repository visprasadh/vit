import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch_size, seq_len, 3, self.heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, dim)
        out = self.proj(out)
        out = self.proj_dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_classes=10,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=0.1,
    ):
        super(VisionTransformer, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout

        # Create patches
        self.patch_dim = (img_size // patch_size) ** 2
        self.patch_embedding = nn.Conv2d(
            3, dim, kernel_size=patch_size, stride=patch_size
        )

        # Positional encoding
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.patch_dim + 1, dim)
        )  # +1 for class token

        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList()
        for _ in range(depth):
            # Implement transformer block with residual connections
            self.transformer_blocks.append(
                nn.ModuleList(
                    [
                        nn.LayerNorm(dim),
                        MultiHeadAttention(dim=dim, heads=heads, dropout=dropout),
                        nn.LayerNorm(dim),
                        FeedForward(dim=dim, hidden_dim=mlp_dim, dropout=dropout),
                    ]
                )
            )

        # MLP head
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, x):
        batch_size = x.shape[0]

        # Create patches
        x = self.patch_embedding(x)  # Shape: (batch_size, dim, h, w)
        x = x.flatten(2).transpose(1, 2)  # Shape: (batch_size, num_patches, dim)

        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional encoding
        x = x + self.position_embedding

        # Pass through transformer blocks with residual connections
        for norm1, attn, norm2, ff in self.transformer_blocks:
            x = x + attn(norm1(x))
            x = x + ff(norm2(x))

        # Use only the class token for classification
        x = x[:, 0]

        # MLP head
        x = self.mlp_head(x)
        return x
