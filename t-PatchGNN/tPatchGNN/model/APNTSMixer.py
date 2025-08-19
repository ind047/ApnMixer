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
            nn.Linear(num_patches * expansion_factor, num_patches),
            nn.Dropout(0.1),
        )

        # Channel mixing: mix across different time series (vital signs)
        self.channel_norm = nn.LayerNorm(d_model)
        self.channel_mlp = nn.Sequential(
            nn.Linear(num_series, num_series * expansion_factor),
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

        # --- True PatchTSMixer Layers ---
        self.mixer_layers = nn.ModuleList(
            [
                AdaptiveMixerBlock(
                    d_model=self.d_model,
                    num_patches=self.n_patches,
                    num_series=self.n_series,  # Added this parameter
                    expansion_factor=2,
                    dropout=0.1,
                )
                for _ in range(args.nlayer)
            ]
        )

        self.norm = nn.LayerNorm(self.d_model)

        # ADD ONLY THIS: Attention aggregation (optional)
        self.use_attention = getattr(args, "use_attention", True)  # Can be toggled
        if self.use_attention:
            self.attention_aggregation = LightweightAttentionAggregation(
                self.d_model, self.n_patches
            )

        # Custom decoder
        # self.decoder = nn.Sequential(
        #     nn.Linear(self.d_model + self.d_te, self.d_model),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.d_model, 1),
        # )
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model + self.d_te, self.d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model // 2, 1),
        )

        # Add learning rate scheduler support
        self.warmup_epochs = 10
        self.max_lr = args.lr

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

        # Apply adaptive mixer layers
        mixer_output = h_p_proj

        for mixer_layer in self.mixer_layers:
            mixer_output = mixer_layer(mixer_output)

        # Apply final normalization
        mixer_output_final = self.norm(mixer_output)

        # ONLY THIS LINE CHANGES: Replace simple mean with attention
        if self.use_attention:
            h_final = self.attention_aggregation(mixer_output_final)  # (B, D, d_model)
        else:
            h_final = mixer_output_final.mean(dim=2)  # Original simple mean

        # Continue with aggregation and decoding
        Lp = t_pred.shape[1]
        h_final_re = h_final.unsqueeze(2).repeat(1, 1, Lp, 1)
        t_pred_re = t_pred.unsqueeze(1).unsqueeze(-1).repeat(1, D, 1, 1)
        te_pred = self.time_embedding(t_pred_re)

        decoder_input = torch.cat([h_final_re, te_pred], dim=-1)
        output = self.decoder(decoder_input).squeeze(-1)

        return output.permute(0, 2, 1).unsqueeze(0)
