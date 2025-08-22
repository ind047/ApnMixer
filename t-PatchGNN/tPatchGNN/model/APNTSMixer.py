import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming APN.py is in the same directory
from .APN import (
    LearnableTimeEmbedding,
    AdaptivePatching,
    WeightedAggregation,
)


class AdaptiveMixerBlock(nn.Module):
    """True PatchTSMixer-style mixer block"""

    def __init__(
        self, d_model, num_patches, num_series, expansion_factor=2, dropout=0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.num_patches = num_patches
        self.num_series = num_series

        # Patch mixing: mix across time patches
        self.patch_norm = nn.LayerNorm(d_model)
        self.patch_mlp = nn.Sequential(
            nn.Linear(num_patches, num_patches * expansion_factor),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(num_patches * expansion_factor, num_patches * expansion_factor),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(num_patches * expansion_factor, num_patches),
            nn.Dropout(0.1),
        )

        # Channel mixing: mix across different time series (vital signs)
        self.channel_norm = nn.LayerNorm(d_model)
        self.channel_mlp = nn.Sequential(
            nn.Linear(num_series, num_series * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_series * expansion_factor, num_series * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_series * expansion_factor, num_series),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x shape: (B, D, P, d_model)
        B, D, P, d_model = x.shape

        # 1. Patch mixing: mix information across time patches
        residual = x
        x = self.patch_norm(x)  # (B, D, P, d_model)
        x = x.permute(0, 1, 3, 2)  # (B, D, d_model, P)
        x = self.patch_mlp(x)  # Mix across P dimension
        x = x.permute(0, 1, 3, 2)  # (B, D, P, d_model)
        x = x + residual  # Residual connection

        # 2. Channel mixing: mix information across different time series
        residual = x
        x = self.channel_norm(x)  # (B, D, P, d_model)
        x = x.permute(0, 2, 3, 1)  # (B, P, d_model, D) - move series to last dim
        x = self.channel_mlp(x)  # Mix across D (series) dimension
        x = x.permute(0, 3, 1, 2)  # (B, D, P, d_model) - back to original
        x = x + residual  # Residual connection

        return x


class AttentionMixerBlock(nn.Module):
    """AdaptiveMixerBlock with attention integrated into patch and channel mixing"""

    def __init__(
        self, d_model, num_patches, num_series, expansion_factor=2, dropout=0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.num_patches = num_patches
        self.num_series = num_series

        # --- Patch Mixing with Attention ---
        self.patch_norm = nn.LayerNorm(d_model)

        # Patch attention: lightweight single-head attention for memory efficiency
        self.patch_attention = nn.MultiheadAttention(
            d_model, num_heads=1, dropout=dropout, batch_first=False
        )

        # Enhanced patch MLP
        self.patch_mlp = nn.Sequential(
            nn.Linear(num_patches, num_patches * expansion_factor),
            nn.LayerNorm(num_patches * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_patches * expansion_factor, num_patches),
            nn.Dropout(dropout),
        )

        # --- Channel Mixing with Attention ---
        self.channel_norm = nn.LayerNorm(d_model)

        # Simplified channel attention: use linear attention for memory efficiency
        self.channel_attention_query = nn.Linear(d_model, d_model // 2)
        self.channel_attention_key = nn.Linear(d_model, d_model // 2)
        self.channel_attention_value = nn.Linear(d_model, d_model)

        # Enhanced channel MLP
        self.channel_mlp = nn.Sequential(
            nn.Linear(num_series, num_series * expansion_factor),
            nn.LayerNorm(num_series * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_series * expansion_factor, num_series),
            nn.Dropout(dropout),
        )

        # Learnable mixing weights for attention + MLP combination
        self.patch_mix_weight = nn.Parameter(torch.tensor(0.5))
        self.channel_mix_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # x shape: (B, D, P, d_model)
        B, D, P, d_model = x.shape

        # === 1. Patch Mixing with Attention ===
        residual = x
        x = self.patch_norm(x)  # (B, D, P, d_model)

        # Apply patch attention across time patches
        x_for_attention = x.permute(2, 0, 1, 3).reshape(
            P, B * D, d_model
        )  # (P, B*D, d_model)
        x_patch_attended, patch_attention_weights = self.patch_attention(
            x_for_attention, x_for_attention, x_for_attention
        )
        x_patch_attended = x_patch_attended.reshape(P, B, D, d_model).permute(
            1, 2, 0, 3
        )  # (B, D, P, d_model)

        # Apply patch MLP
        x_patch_mlp = x.permute(0, 1, 3, 2)  # (B, D, d_model, P)
        x_patch_mlp = self.patch_mlp(x_patch_mlp)  # Mix across P dimension
        x_patch_mlp = x_patch_mlp.permute(0, 1, 3, 2)  # (B, D, P, d_model)

        # Combine attention and MLP outputs with learnable weight
        patch_alpha = torch.sigmoid(self.patch_mix_weight)
        x_patch_combined = (
            patch_alpha * x_patch_attended + (1 - patch_alpha) * x_patch_mlp
        )

        x = residual + x_patch_combined

        # === 2. Channel Mixing with Attention ===
        residual = x
        x = self.channel_norm(x)  # (B, D, P, d_model)

        # Apply simplified channel attention (process each patch separately for memory)
        channel_attended_list = []
        for p in range(P):
            x_p = x[:, :, p, :]  # (B, D, d_model) - single patch position

            # Compute queries, keys, values
            Q = self.channel_attention_query(x_p)  # (B, D, d_model//2)
            K = self.channel_attention_key(x_p)  # (B, D, d_model//2)
            V = self.channel_attention_value(x_p)  # (B, D, d_model)

            # Compute attention scores
            scores = (
                torch.matmul(Q, K.transpose(-2, -1)) / (d_model // 2) ** 0.5
            )  # (B, D, D)
            attn_weights = torch.softmax(scores, dim=-1)  # (B, D, D)

            # Apply attention
            attended = torch.matmul(attn_weights, V)  # (B, D, d_model)
            channel_attended_list.append(attended)

        x_channel_attended = torch.stack(
            channel_attended_list, dim=2
        )  # (B, D, P, d_model)

        # Apply channel MLP
        x_channel_mlp = x.permute(0, 2, 3, 1)  # (B, P, d_model, D)
        x_channel_mlp = self.channel_mlp(x_channel_mlp)  # Mix across D dimension
        x_channel_mlp = x_channel_mlp.permute(0, 3, 1, 2)  # (B, D, P, d_model)

        # Combine attention and MLP outputs with learnable weight
        channel_alpha = torch.sigmoid(self.channel_mix_weight)
        x_channel_combined = (
            channel_alpha * x_channel_attended + (1 - channel_alpha) * x_channel_mlp
        )

        x = residual + x_channel_combined

        return x, patch_attention_weights


class LightweightAttentionAggregation(nn.Module):
    """Lightweight attention for patch aggregation - doesn't change core architecture"""

    def __init__(self, d_model, n_patches):
        super().__init__()
        # Simple attention mechanism
        self.attention_weights = nn.Linear(d_model, 1)
        self.patch_pos_embed = nn.Parameter(torch.randn(n_patches, d_model) * 0.1)

    def forward(self, h_p):
        # h_p: (B, D, P, d_model)
        B, D, P, d_model = h_p.shape

        # Add positional embeddings
        h_p = h_p + self.patch_pos_embed.unsqueeze(0).unsqueeze(0)

        # Compute attention scores for each patch
        attention_scores = self.attention_weights(h_p)  # (B, D, P, 1)
        attention_weights = torch.softmax(attention_scores, dim=2)  # (B, D, P, 1)

        # Weighted sum instead of mean
        h_final = (h_p * attention_weights).sum(dim=2)  # (B, D, d_model)

        return h_final


class APNTSMixer(nn.Module):
    def __init__(self, args):
        super(APNTSMixer, self).__init__()
        self.n_patches = args.npatch
        self.d_model = args.hid_dim
        self.d_te = args.te_dim
        self.n_series = args.ndim
        self.t_obs = args.t_obs

        # --- tAPN Components ---
        self.time_embedding = LearnableTimeEmbedding(self.d_te)
        self.adaptive_patching = AdaptivePatching(
            self.n_series, self.n_patches, self.t_obs
        )
        self.weighted_aggregation = WeightedAggregation(self.n_series)

        # Projection to match mixer input expectations
        self.projection = nn.Linear(1 + self.d_te, self.d_model)

        # --- Attention Mixer Layers ---
        self.use_attention = getattr(args, "use_attention", True)

        if self.use_attention:
            self.mixer_layers = nn.ModuleList(
                [
                    AttentionMixerBlock(
                        d_model=self.d_model,
                        num_patches=self.n_patches,
                        num_series=self.n_series,
                        expansion_factor=getattr(args, "expansion_factor", 2),
                        dropout=getattr(args, "dropout", 0.1),
                    )
                    for _ in range(args.nlayer)
                ]
            )
        else:
            # Fall back to original mixer blocks
            self.mixer_layers = nn.ModuleList(
                [
                    AdaptiveMixerBlock(
                        d_model=self.d_model,
                        num_patches=self.n_patches,
                        num_series=self.n_series,
                        expansion_factor=2,
                        dropout=0.1,
                    )
                    for _ in range(args.nlayer)
                ]
            )

        self.norm = nn.LayerNorm(self.d_model)

        # Optional end-layer attention for aggregation
        self.use_end_attention = getattr(args, "use_end_attention", False)
        if self.use_end_attention:
            self.attention_aggregation = LightweightAttentionAggregation(
                self.d_model, self.n_patches
            )

        # Enhanced decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model + self.d_te, self.d_model),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.d_model),
            nn.Dropout(0.1),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.d_model // 2),
            nn.Linear(self.d_model // 2, 1),
        )

    def forecasting(self, t_pred, x, t, mask):
        # Same preprocessing as before...
        if len(x.shape) == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        if len(t.shape) == 4 and t.shape[1] == 1:
            t = t.squeeze(1)
            t = t[:, :, 0]
        if len(mask.shape) == 4 and mask.shape[1] == 1:
            mask = mask.squeeze(1)
        if len(t_pred.shape) == 3 and t_pred.shape[1] == 1:
            t_pred = t_pred.squeeze(1)

        B, L, D = x.shape

        # APN processing (same as before)
        x_reshaped = x.permute(0, 2, 1).unsqueeze(-1)
        t_expanded = t.unsqueeze(-1).expand(B, L, D).permute(0, 2, 1).unsqueeze(-1)
        mask_reshaped = mask.permute(0, 2, 1).unsqueeze(-1)

        x_masked = x_reshaped * mask_reshaped
        te = self.time_embedding(t_expanded)
        x_aug = torch.cat([x_masked, te], dim=-1)

        t_left, t_right = self.adaptive_patching()
        h_p = self.weighted_aggregation(x_aug, t_expanded, t_left, t_right)
        h_p_proj = self.projection(h_p)  # (B, D, P, d_model)

        # Apply attention mixer layers
        mixer_output = h_p_proj
        patch_attention_weights_list = []

        if self.use_attention:
            for mixer_layer in self.mixer_layers:
                mixer_output, patch_attn = mixer_layer(mixer_output)
                patch_attention_weights_list.append(patch_attn)
        else:
            for mixer_layer in self.mixer_layers:
                mixer_output = mixer_layer(mixer_output)

        # Apply final normalization
        mixer_output_final = self.norm(mixer_output)

        # Final aggregation
        if self.use_end_attention:
            h_final = self.attention_aggregation(mixer_output_final)  # (B, D, d_model)
        else:
            h_final = mixer_output_final.mean(dim=2)  # Simple mean

        # Continue with decoding
        Lp = t_pred.shape[1]
        h_final_re = h_final.unsqueeze(2).repeat(1, 1, Lp, 1)
        t_pred_re = t_pred.unsqueeze(1).unsqueeze(-1).repeat(1, D, 1, 1)
        te_pred = self.time_embedding(t_pred_re)

        decoder_input = torch.cat([h_final_re, te_pred], dim=-1)
        output = self.decoder(decoder_input).squeeze(-1)

        return output.permute(0, 2, 1).unsqueeze(0)

    def get_attention_weights(self):
        """Utility method to extract attention weights for visualization"""
        if hasattr(self, "patch_attention_weights_list"):
            return {
                "patch_attention": self.patch_attention_weights_list,
            }
        return None