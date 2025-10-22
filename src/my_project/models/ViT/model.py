import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ====================================================================
# COMPLETE ViT MAGNITUDE ESTIMATION MODEL
# Based on Figure 1c from the paper
# ====================================================================


class SelfAttention(nn.Module):
    """Standard Self-Attention block (Figure 1h)"""

    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(
            self.embed_dim
        )
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        attention_output = torch.matmul(attention_weights, V)

        return attention_output


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention (Figure 1g)"""

    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.attention_heads = nn.ModuleList(
            [SelfAttention(embed_dim, dropout) for _ in range(num_heads)]
        )

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention_outputs = [head(x) for head in self.attention_heads]
        output = torch.stack(attention_outputs, dim=0).mean(dim=0)
        output = self.out_proj(output)
        output = self.dropout(output)
        return output


class MLP(nn.Module):
    """MLP block with GELU activation"""

    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerEncoder(nn.Module):
    """Transformer Encoder Block (Figure 1f)"""

    def __init__(self, embed_dim, num_heads=4, mlp_hidden_dim=200, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.msa = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.mlp = MLP(embed_dim, mlp_hidden_dim, dropout)

    def forward(self, x):
        # Norm -> MSA -> Residual
        residual = x
        x = self.norm1(x)
        x = self.msa(x)
        x = x + residual

        # Norm -> MLP -> Residual
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual

        return x


class ConvBlock(nn.Module):
    """Convolution block with linear activation to preserve amplitude"""

    def __init__(
        self, in_channels, out_channels, kernel_size=3, pool_size=2, dropout=0.1
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.MaxPool1d(kernel_size=pool_size)

    def forward(self, x):
        x = self.conv(x)  # Linear activation (no non-linearity)
        x = self.dropout(x)
        x = self.pool(x)
        return x


class PatchEmbedding(nn.Module):
    """Patch embedding with positional encoding"""

    def __init__(self, input_dim=75, num_channels=32, patch_size=5, embed_dim=100):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = input_dim // patch_size
        patch_dim = num_channels * patch_size

        self.projection = nn.Linear(patch_dim, embed_dim)
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim)
        )

    def forward(self, x):
        batch_size = x.shape[0]

        # Create patches
        x = x.unfold(dimension=2, size=self.patch_size, step=self.patch_size)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size, self.num_patches, -1)

        # Apply embedding and add positional encoding
        x = self.projection(x)
        x = x + self.position_embedding

        return x


class ViTMagnitudeEstimator(nn.Module):
    """
    Complete ViT Magnitude Estimation Model (Figure 1c)

    Architecture from paper:
    - Input: 30s, 3-component seismograms (3000 samples @ 100Hz)
    - 4 Convolution blocks with pooling rates [2, 2, 2, 5]
    - Patch size: 5, Num patches: 15
    - Projection dimension: 100
    - 4 Transformer encoders with 4 attention heads
    - Final MLP: [1000, 500] neurons
    - Output: Single magnitude value
    """

    def __init__(
        self,
        input_length=3000,
        input_channels=3,
        conv_channels=[64, 32, 32, 32],
        kernel_size=3,
        pool_sizes=[2, 2, 2, 5],
        patch_size=5,
        embed_dim=100,
        num_transformer_blocks=4,
        num_heads=4,
        transformer_mlp_dim=200,
        final_mlp_dims=[1000, 500],
        dropout=0.1,
        final_dropout=0.5,
    ):
        super().__init__()

        # ============================================================
        # CONVOLUTIONAL FEATURE EXTRACTION (Block A in Figure 1c)
        # ============================================================
        self.conv_blocks = nn.ModuleList()
        in_ch = input_channels

        for out_ch, pool in zip(conv_channels, pool_sizes):
            self.conv_blocks.append(
                ConvBlock(in_ch, out_ch, kernel_size, pool, dropout)
            )
            in_ch = out_ch

        # Calculate output dimension after convolutions
        # 3000 -> 1500 -> 750 -> 375 -> 75
        conv_output_dim = input_length
        for pool in pool_sizes:
            conv_output_dim = conv_output_dim // pool

        # ============================================================
        # PATCH EMBEDDING
        # ============================================================
        self.patch_embed = PatchEmbedding(
            input_dim=conv_output_dim,
            num_channels=conv_channels[-1],
            patch_size=patch_size,
            embed_dim=embed_dim,
        )

        # ============================================================
        # TRANSFORMER ENCODERS (repeated U times, U=4)
        # ============================================================
        self.transformer_encoders = nn.ModuleList(
            [
                TransformerEncoder(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_hidden_dim=transformer_mlp_dim,
                    dropout=dropout,
                )
                for _ in range(num_transformer_blocks)
            ]
        )

        # ============================================================
        # FINAL MLP FOR MAGNITUDE PREDICTION
        # ============================================================
        num_patches = self.patch_embed.num_patches
        flatten_dim = num_patches * embed_dim

        mlp_layers = []
        in_dim = flatten_dim

        for hidden_dim in final_mlp_dims:
            mlp_layers.append(nn.Linear(in_dim, hidden_dim))
            mlp_layers.append(nn.GELU())
            mlp_layers.append(nn.Dropout(final_dropout))
            in_dim = hidden_dim

        self.final_mlp = nn.Sequential(*mlp_layers)

        # Output layer: single neuron with linear activation for magnitude
        self.output = nn.Linear(final_mlp_dims[-1], 1)

    def forward(self, x):
        """
        Args:
            x: (batch, 3, 3000) - 3-component seismograms
        Returns:
            magnitude: (batch, 1) - estimated magnitude
        """
        # Convolutional feature extraction
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        # x is now (batch, 32, 75)

        # Patch embedding
        x = self.patch_embed(x)
        # x is now (batch, 15, 100)

        # Transformer encoders
        for transformer in self.transformer_encoders:
            x = transformer(x)
        # x is still (batch, 15, 100)

        # Flatten for final MLP
        x = x.reshape(x.shape[0], -1)
        # x is now (batch, 1500)

        # Final MLP
        x = self.final_mlp(x)

        # Output magnitude
        magnitude = self.output(x)

        return magnitude


# ====================================================================
# MODEL INSTANTIATION AND TESTING
# ====================================================================
def test_model():
    print("=" * 70)
    print("Testing ViT Magnitude Estimation Model")
    print("=" * 70)

    # Create model with paper specifications
    model = ViTMagnitudeEstimator(
        input_length=3000,
        input_channels=3,
        conv_channels=[64, 32, 32, 32],
        kernel_size=3,
        pool_sizes=[2, 2, 2, 5],
        patch_size=5,
        embed_dim=100,
        num_transformer_blocks=4,
        num_heads=4,
        transformer_mlp_dim=200,
        final_mlp_dims=[1000, 500],
        dropout=0.1,
        final_dropout=0.5,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Architecture:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Paper reports: 2,840,825 parameters")

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 3000)

    print(f"\nTesting forward pass:")
    print(f"  Input shape: {x.shape}")

    model.eval()
    with torch.no_grad():
        output = model(x)

    print(f"  Output shape: {output.shape}")
    print(f"  Expected: ({batch_size}, 1)")
    print(f"  Sample predictions: {output.squeeze().numpy()}")

    print("\n" + "=" * 70)
    print("✓ Model created successfully!")
    print("=" * 70)

    return model


if __name__ == "__main__":
    model = test_model()
