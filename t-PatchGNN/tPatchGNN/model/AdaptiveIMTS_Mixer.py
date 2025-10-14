"""
AdaptiveIMTS_Mixer: A hybrid model combining APN's adaptive patching with IMTS_Mixer's mixing capabilities.

This model features:
- APN-style adaptive patching with per-series learnable boundaries
- APN-style weighted aggregation with sigmoid weighting
- IMTS_Mixer-style mixer blocks with multi-head attention
- Time-aware encoding and forecasting
"""

import torch
#this is to just make changes
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / norm * self.gamma


class LearnableTimeEmbedding(nn.Module):
    def __init__(self, d_model):
        super(LearnableTimeEmbedding, self).__init__()
        self.d_model = d_model
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, d_model - 1)

    def forward(self, tt):
        # tt: (B, N, L, 1)
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)


class AdaptivePatching(nn.Module):
    """
    APN-style adaptive patching with per-series learnable boundaries.
    Each series learns its own patch boundaries using delta and lambda parameters.
    """

    def __init__(self, n_series, n_patches, t_obs):
        super(AdaptivePatching, self).__init__()
        self.n_series = n_series
        self.n_patches = n_patches
        self.t_obs = t_obs

        # Each series learns its own patch boundaries - like APN
        self.delta = nn.Parameter(
            torch.randn(n_series, n_patches)
        )  # (n_series, n_patches)
        self.lamb = nn.Parameter(
            torch.randn(n_series, n_patches)
        )  # (n_series, n_patches)

    def forward(self):
        s_init = self.t_obs / self.n_patches
        c_p = (torch.arange(self.n_patches, device=self.delta.device) + 0.5) * s_init

        t_left = c_p.unsqueeze(0) - s_init / 2 + self.delta  # (n_series, n_patches)
        t_right = t_left + torch.exp(self.lamb) * s_init  # (n_series, n_patches)

        return t_left, t_right


class WeightedAggregation(nn.Module):
    """
    APN-style weighted aggregation using sigmoid weighting functions.
    Aggregates observations within patches using learned kappa parameter.
    """

    def __init__(self, n_series):
        super(WeightedAggregation, self).__init__()
        self.kappa = nn.Parameter(torch.randn(n_series, 1))

    def forward(self, x_aug, t, t_left, t_right):
        # x_aug: (B, N, L, F)
        # t: (B, N, L, 1)
        # t_left, t_right: (N, P)

        B, N, L, n_features = x_aug.shape

        t_left = t_left.unsqueeze(0).unsqueeze(2)  # (1, N, 1, P)
        t_right = t_right.unsqueeze(0).unsqueeze(2)  # (1, N, 1, P)

        # Adjust kappa to match actual data dimensions if needed
        if self.kappa.shape[0] != N:
            if self.kappa.shape[0] > N:
                kappa = (
                    F.softplus(self.kappa[:N, :]).unsqueeze(0).unsqueeze(2)
                )  # (1, N, 1, 1)
            else:
                # Repeat the last kappa value for missing dimensions
                repeat_count = N - self.kappa.shape[0]
                kappa_extended = torch.cat(
                    [self.kappa, self.kappa[-1:, :].repeat(repeat_count, 1)], dim=0
                )
                kappa = (
                    F.softplus(kappa_extended).unsqueeze(0).unsqueeze(2)
                )  # (1, N, 1, 1)
        else:
            kappa = F.softplus(self.kappa).unsqueeze(0).unsqueeze(2)  # (1, N, 1, 1)

        # alpha: (B, N, L, P)
        alpha = torch.sigmoid((t_right - t) / kappa) * torch.sigmoid(
            (t - t_left) / kappa
        )

        # h_p: (B, N, P, F) - weighted aggregation over time dimension L
        # Expand dimensions for broadcasting: alpha(B,N,L,P,1) * x_aug(B,N,L,1,F) -> (B,N,L,P,F)
        alpha_expanded = alpha.unsqueeze(-1)  # (B, N, L, P, 1)
        x_aug_expanded = x_aug.unsqueeze(-2)  # (B, N, L, 1, F)

        # Weighted features: (B, N, L, P, F)
        weighted = alpha_expanded * x_aug_expanded

        # Sum over time dimension L: (B, N, L, P, F) -> (B, N, P, F)
        numerator = weighted.sum(dim=2)

        # Normalization: sum of alpha over time: (B, N, L, P) -> (B, N, P)
        denominator = alpha.sum(dim=2)  # (B, N, P)
        denominator = denominator.unsqueeze(-1)  # (B, N, P, 1) for broadcasting with F

        h_p = numerator / (denominator + 1e-8)  # (B, N, P, F)

        return h_p


class QueryBasedAggregation(nn.Module):
    """
    APN-style query-based aggregation with learnable query and positional encoding.
    """

    def __init__(self, d_model, n_patches):
        super(QueryBasedAggregation, self).__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.pe = nn.Parameter(torch.randn(1, n_patches, d_model))

    def forward(self, h_p):
        # h_p: (B, N, P, D)
        B, N, P, D = h_p.shape

        h_pe = h_p + self.pe  # Add positional encoding

        # (B, N, P, D) * (1, 1, 1, D) -> (B, N, P, D) -> (B, N, P)
        s = torch.sum(h_pe * self.query, dim=-1) / (D**0.5)
        beta = F.softmax(s, dim=-1)  # (B, N, P)

        # (B, N, P, D) * (B, N, P, 1) -> (B, N, D)
        h_final = torch.sum(h_pe * beta.unsqueeze(-1), dim=2)

        return h_final


class MixerBlock(nn.Module):
    """
    IMTS_Mixer-style mixer block with multi-head attention for channel mixing
    and MLP for hidden dimension mixing.
    """

    def __init__(self, d_model, n_channels, n_heads=4):
        super().__init__()
        # Multi-head attention for channel mixing (IMTS_Mixer style)
        # self.channel_attention = nn.MultiheadAttention(
        #     embed_dim=d_model, num_heads=n_heads, batch_first=True
        # )
        # self.channel_norm = RMSNorm(n_channels)  # Normalize across channels
        
        self.channel_norm_old = RMSNorm(n_channels)  # Normalize across channels
        self.channel_mlp = nn.Sequential(
            nn.Linear(n_channels, n_channels), nn.ReLU(inplace=True)
        )

        # Hidden dimension mixing (IMTS_Mixer style)
        self.hidden_norm = RMSNorm(d_model)  # Normalize across hidden features
        self.hidden_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: (B, N, D) - batch, channels, features
        B, N, D = x.shape

        # Channel mixing using multi-head attention
        # residual = x

        # # Handle dynamic channel dimensions for normalization
        # x_permuted = x.permute(0, 2, 1)  # (B, D, N)
        # if self.channel_norm.d_model != N:
        #     # Apply normalization manually if dimensions don't match
        #     norm = torch.sqrt(
        #         torch.mean(x_permuted**2, dim=-1, keepdim=True) + self.channel_norm.eps
        #     )
        #     x_norm = x_permuted / norm
        #     # We can't use the learned gamma parameter if dimensions don't match
        #     x_norm = x_norm.permute(0, 2, 1)  # (B, N, D)
        # else:
        #     x_norm = self.channel_norm(x_permuted).permute(0, 2, 1)  # (B, N, D)

        # # Apply multi-head attention across channels
        # attn_output, _ = self.channel_attention(
        #     query=x_norm,  # (B, N, D)
        #     key=x_norm,  # (B, N, D)
        #     value=x_norm,  # (B, N, D)
        # )
        # x = attn_output + residual  # Residual connection
        
        residual = x
        x = self.channel_norm_old(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.channel_mlp(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + residual

        # Hidden mixing
        residual = x
        x = self.hidden_norm(x)
        x = self.hidden_mlp(x)
        x = x + residual
        return x


class AdaptiveIMTS_Mixer(nn.Module):
    """
    AdaptiveIMTS_Mixer: Hybrid model combining APN's adaptive patching with IMTS_Mixer's mixing capabilities.

    Features:
    - APN-style adaptive patching with per-series learnable boundaries
    - APN-style weighted aggregation with sigmoid weighting
    - IMTS_Mixer-style mixer blocks with multi-head attention
    - Time-aware encoding and forecasting
    """

    def __init__(self, args):
        super(AdaptiveIMTS_Mixer, self).__init__()

        self.n_patches = args.npatch
        self.d_model = args.hid_dim
        self.d_te = args.te_dim
        self.n_series = args.ndim
        self.t_obs = args.t_obs  # Total observation window size
        self.n_layers = args.nlayer
        self.d_out = args.d_out if hasattr(args, "d_out") else self.d_model
        self.n_heads = getattr(args, "n_heads", 4)  # Default 4 heads

        # APN-style components
        self.time_embedding = LearnableTimeEmbedding(self.d_te)
        self.adaptive_patching = AdaptivePatching(
            self.n_series, self.n_patches, self.t_obs
        )
        self.weighted_aggregation = WeightedAggregation(self.n_series)
        self.projection = nn.Linear(1 + self.d_te, self.d_model)
        self.query_aggregation = QueryBasedAggregation(self.d_model, self.n_patches)

        # IMTS_Mixer-style components
        self.channel_bias = nn.Parameter(torch.randn(1, self.n_series, self.d_model))

        self.mixer_blocks = nn.ModuleList(
            [
                MixerBlock(self.d_model, self.n_series, self.n_heads)
                for _ in range(self.n_layers)
            ]
        )

        if self.d_model != self.d_out:
            self.out_proj = nn.Linear(self.d_model, self.d_out)
        else:
            self.out_proj = nn.Identity()

        self.decoder = nn.Sequential(
            nn.Linear(self.d_out + self.d_te, self.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model, 1),
        )

        self.args = args

    def forecasting(self, t_pred, x, t, mask):
        """
        APN-style forecasting method.

        Args:
            t_pred: Prediction timestamps (B, Lp)
            x: Observed data (B, L, D)
            t: Observed timestamps (B, L)
            mask: Observation mask (B, L, D)

        Returns:
            output: Predictions (1, B, Lp, D)
        """
        B, L, D = x.shape

        # Expand time to match data dimensions: (B, L) -> (B, L, D)
        t_expanded = t.unsqueeze(-1).expand(B, L, D)

        # Reshape for APN processing: (B, L, D) -> (B, D, L, 1)
        x = x.permute(0, 2, 1).unsqueeze(-1)  # (B, D, L, 1)
        t_expanded = t_expanded.permute(0, 2, 1).unsqueeze(-1)  # (B, D, L, 1)
        mask = mask.permute(0, 2, 1).unsqueeze(-1)  # (B, D, L, 1)

        # Apply mask to x first, then add time embedding
        x = x * mask  # Apply mask to data only
        te = self.time_embedding(t_expanded)  # (B, D, L, d_te)
        x_aug = torch.cat([x, te], dim=-1)  # (B, D, L, 1+d_te)

        # APN-style adaptive patching and weighted aggregation
        t_left, t_right = self.adaptive_patching()  # (n_series, P)

        # Adjust patching parameters to match actual data dimensions if needed
        if t_left.shape[0] != D:
            if t_left.shape[0] > D:
                t_left = t_left[:D, :]
                t_right = t_right[:D, :]
            else:
                # Repeat the last series parameters for missing dimensions
                repeat_count = D - t_left.shape[0]
                t_left = torch.cat(
                    [t_left, t_left[-1:, :].repeat(repeat_count, 1)], dim=0
                )
                t_right = torch.cat(
                    [t_right, t_right[-1:, :].repeat(repeat_count, 1)], dim=0
                )

        h_p = self.weighted_aggregation(
            x_aug, t_expanded, t_left, t_right
        )  # (B, D, P, 1+d_te)
        h_p = self.projection(h_p)  # (B, D, P, d_model)

        #h_final = self.query_aggregation(h_p)  # (B, D, d_model)
        h_final = torch.sum(h_p, dim=2)  # (B, D, d_model)
        # Handle unobserved channels
        # Ensure mask dimensions match h_final dimensions
        B, D, d_model = h_final.shape
        if mask.shape[1] != D:
            # Adjust mask to match actual data dimensions
            if mask.shape[1] > D:
                mask_adjusted = mask[:, :D]
            else:
                # Pad mask with zeros for additional dimensions
                pad_size = D - mask.shape[2]
                mask_adjusted = torch.cat(
                    [mask, torch.zeros(B, mask.shape[1], pad_size, device=mask.device)],
                    dim=2,
                )
        else:
            mask_adjusted = mask

        unobserved_mask = (mask_adjusted.sum(dim=2) == 0).unsqueeze(-1)  # (B, D, 1)

        # # Dynamically adjust channel_bias to match actual data dimensions
        # if self.channel_bias.shape[1] != D:
        #     # Create or adjust channel_bias to match current data dimensions
        #     if self.channel_bias.shape[1] > D:
        #         channel_bias = self.channel_bias[:, :D, :]
        #     else:
        #         repeat_count = D - self.channel_bias.shape[1]
        #         channel_bias = torch.cat(
        #             [
        #                 self.channel_bias,
        #                 self.channel_bias[:, -1:, :].repeat(1, repeat_count, 1),
        #             ],
        #             dim=1,
        #         )
        # else:
        channel_bias = self.channel_bias

        # Ensure dimensional compatibility before operations
        # Handle case where any tensor dimensions don't match
        h_shape = h_final.shape  # (B, D, d_model)
        mask_shape = unobserved_mask.shape  # (B, D, 1)
        bias_shape = channel_bias.shape  # (1, D, d_model)

        # Force all tensors to have the same D dimension
        # actual_D = h_shape[1]
        # if mask_shape[1] != actual_D:
        #     if mask_shape[1] > actual_D:
        #         unobserved_mask = unobserved_mask[:, :actual_D, :]
        #     else:
        #         pad_D = actual_D - mask_shape[1]
        #         pad_tensor = torch.zeros(
        #             mask_shape[0], pad_D, mask_shape[2], device=unobserved_mask.device
        #         )
        #         unobserved_mask = torch.cat([unobserved_mask, pad_tensor], dim=1)

        # if bias_shape[1] != actual_D:
        #     if bias_shape[1] > actual_D:
        #         channel_bias = channel_bias[:, :actual_D, :]
        #     else:
        #         pad_D = actual_D - bias_shape[1]
        #         pad_tensor = channel_bias[:, -1:, :].repeat(1, pad_D, 1)
        #         channel_bias = torch.cat([channel_bias, pad_tensor], dim=1)

        # h_final = (
        #     h_final * (1 - unobserved_mask.squeeze(3).float())
        #     + channel_bias * unobserved_mask.squeeze(3).float()
        # )
        h_final = h_final + channel_bias

        # Apply IMTS_Mixer-style mixer blocks
        for mixer_block in self.mixer_blocks:
            h_final = mixer_block(h_final)

        h_final = self.out_proj(h_final)

        # Decoder
        Lp = t_pred.shape[1]
        h_final_re = h_final.unsqueeze(2).repeat(1, 1, Lp, 1)  # (B, D, Lp, d_out)

        t_pred_re = (
            t_pred.unsqueeze(1).unsqueeze(-1).repeat(1, D, 1, 1)
        )  # (B, D, Lp, 1)
        te_pred = self.time_embedding(t_pred_re)  # (B, D, Lp, d_te)

        decoder_input = torch.cat([h_final_re, te_pred], dim=-1)

        output = self.decoder(decoder_input).squeeze(-1)  # (B, D, Lp)

        return output.permute(0, 2, 1).unsqueeze(0)  # (1, B, Lp, D)

    def get_patch_info(self):
        """
        Returns information about learned patch boundaries for each series.
        Useful for analysis and visualization.
        """
        with torch.no_grad():
            t_left, t_right = self.adaptive_patching()

            return {
                "t_left": t_left.cpu().numpy(),  # (n_series, n_patches)
                "t_right": t_right.cpu().numpy(),  # (n_series, n_patches)
                "delta": self.adaptive_patching.delta.cpu().numpy(),
                "lambda": self.adaptive_patching.lamb.cpu().numpy(),
                "n_series": self.n_series,
                "n_patches": self.n_patches,
            }
