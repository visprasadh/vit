import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads, dropout=0.1, lora_alpha=1, lora_r=8):
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(dropout)
        self.lora_alpha = lora_alpha
        self.lora_r = lora_r

    def forward(self, x, lora_A=None, lora_B=None):
        batch_size, seq_len, dim = x.shape

        # Apply standard linear transformation
        qkv_output = self.qkv(x)

        # Add LoRA contribution if enabled
        if lora_A is not None and lora_B is not None:
            # LoRA: x -> A -> B -> output, with appropriate scaling
            # Correct application:
            # 1. x @ A.T, then the result @ B.T
            # 2. Apply scaling factor alpha/r
            lora_output = x @ lora_A.transpose(-1, -2) @ lora_B.transpose(-1, -2)
            # Apply scaling factor
            lora_output = lora_output * (self.lora_alpha / self.lora_r)
            qkv_output = qkv_output + lora_output

        qkv = qkv_output.reshape(
            batch_size, seq_len, 3, self.heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
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
        lora_alpha=16,
        lora_r=8,
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
        self.lora_alpha = lora_alpha
        self.lora_r = lora_r

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
                        MultiHeadAttention(
                            dim=dim,
                            heads=heads,
                            dropout=dropout,
                            lora_alpha=lora_alpha,
                            lora_r=lora_r,
                        ),
                        nn.LayerNorm(dim),
                        FeedForward(
                            dim=dim,
                            hidden_dim=mlp_dim,
                            dropout=dropout,
                        ),
                    ]
                )
            )

        # MLP head
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, lora_A=None, lora_B=None):
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
        for i, (norm1, attn, norm2, ff) in enumerate(self.transformer_blocks):
            if lora_A is not None and lora_B is not None:
                x = x + attn(norm1(x), lora_A[i], lora_B[i])
            else:
                x = x + attn(norm1(x))
            x = x + ff(norm2(x))

        # Use only the class token for classification
        x = x[:, 0]

        # MLP head
        x = self.mlp_head(x)
        return x
