import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / norm * self.gamma


class ObservationEncoder(nn.Module):
    def __init__(self, d_model, d_time):
        super(ObservationEncoder, self).__init__()
        self.value_encoder = nn.Linear(1, d_model)
        self.time_encoder = nn.Sequential(
            nn.Linear(1, d_time), nn.ReLU(inplace=True), nn.Linear(d_time, d_model)
        )

    def forward(self, v, t):
        v_enc = self.value_encoder(v)
        t_enc = self.time_encoder(t)
        return v_enc * t_enc


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
    def __init__(self, n_series, n_patches, t_obs):
        super(AdaptivePatching, self).__init__()
        self.n_series = n_series
        self.n_patches = n_patches
        self.t_obs = t_obs

        # Each series learns its own patch boundaries
        self.delta = nn.Parameter(torch.randn(n_series, n_patches))
        self.lamb = nn.Parameter(torch.randn(n_series, n_patches))

    def forward(self):
        s_init = self.t_obs / self.n_patches
        c_p = (torch.arange(self.n_patches, device=self.delta.device) + 0.5) * s_init

        t_left = c_p.unsqueeze(0) - s_init / 2 + self.delta  # (n_series, n_patches)
        t_right = t_left + torch.exp(self.lamb) * s_init  # (n_series, n_patches)

        return t_left, t_right


class AdaptiveWeightedAggregation(nn.Module):
    def __init__(self, d_model, d_time, n_series):
        super(AdaptiveWeightedAggregation, self).__init__()
        self.observation_encoder = ObservationEncoder(d_model, d_time)
        self.kappa = nn.Parameter(torch.randn(n_series, 1))
        self.d_model = d_model

    def forward(self, v, t, mask, t_left, t_right):
        # v: (B, N, L, 1), t: (B, N, L, 1), mask: (B, N, L, 1)
        # t_left, t_right: (N, P)

        B, N, L, _ = v.shape
        P = t_left.shape[1]

        # Encode observations
        h = self.observation_encoder(v, t)  # (B, N, L, d_model)

        # Expand patch boundaries for broadcasting
        t_left = t_left.unsqueeze(0).unsqueeze(2)  # (1, N, 1, P)
        t_right = t_right.unsqueeze(0).unsqueeze(2)  # (1, N, 1, P)

        kappa = F.softplus(self.kappa).unsqueeze(0).unsqueeze(2)  # (1, N, 1, 1)

        # Compute adaptive weights: (B, N, L, P)
        alpha = torch.sigmoid((t_right - t) / kappa) * torch.sigmoid(
            (t - t_left) / kappa
        )

        # Apply observation mask
        alpha = alpha * mask  # (B, N, L, P)

        # Weighted aggregation over time dimension
        alpha_expanded = alpha.unsqueeze(-1)  # (B, N, L, P, 1)
        h_expanded = h.unsqueeze(-2)  # (B, N, L, 1, d_model)

        # Weighted features: (B, N, L, P, d_model)
        weighted = alpha_expanded * h_expanded

        # Sum over time dimension: (B, N, P, d_model)
        numerator = weighted.sum(dim=2)
        denominator = alpha.sum(dim=2).unsqueeze(-1)  # (B, N, P, 1)

        h_p = numerator / (denominator + 1e-8)  # (B, N, P, d_model)

        return h_p


class QueryBasedPatchAggregation(nn.Module):
    def __init__(self, d_model, n_patches):
        super(QueryBasedPatchAggregation, self).__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.pe = nn.Parameter(torch.randn(1, n_patches, d_model))

    def forward(self, h_p):
        # h_p: (B, N, P, d_model)
        B, N, P, D = h_p.shape

        h_pe = h_p + self.pe  # Add positional encoding

        # Attention mechanism
        s = torch.sum(h_pe * self.query, dim=-1) / (D**0.5)  # (B, N, P)
        beta = F.softmax(s, dim=-1)  # (B, N, P)

        # Final aggregation: (B, N, d_model)
        h_final = torch.sum(h_pe * beta.unsqueeze(-1), dim=2)

        return h_final


class MixerBlock(nn.Module):
    def __init__(self, d_model, n_channels, n_heads=4):
        super().__init__()
        # Multi-head attention for channel mixing
        self.channel_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True
        )
        self.channel_norm = RMSNorm(n_channels)

        # Feature mixing
        self.hidden_norm = RMSNorm(d_model)
        self.hidden_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Channel mixing using multi-head attention
        residual = x
        x_norm = self.channel_norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        attn_output, _ = self.channel_attention(query=x_norm, key=x_norm, value=x_norm)
        x = attn_output + residual

        # Feature mixing
        residual = x
        x = self.hidden_norm(x)
        x = self.hidden_mlp(x)
        x = x + residual

        return x


class AdaptiveIMTS_Mixer(nn.Module):
    def __init__(self, args):
        super(AdaptiveIMTS_Mixer, self).__init__()
        self.n_channels = args.ndim
        self.d_model = args.hid_dim
        self.d_time = args.te_dim
        self.n_layers = args.nlayer
        self.n_patches = args.npatch
        self.t_obs = args.t_obs
        self.d_out = args.d_out if hasattr(args, "d_out") else self.d_model
        self.n_heads = args.n_heads if hasattr(args, "n_heads") else 4

        # Adaptive patching components (from APN)
        self.adaptive_patching = AdaptivePatching(
            self.n_channels, self.n_patches, self.t_obs
        )
        self.adaptive_aggregation = AdaptiveWeightedAggregation(
            self.d_model, self.d_time, self.n_channels
        )
        self.query_aggregation = QueryBasedPatchAggregation(
            self.d_model, self.n_patches
        )

        # Channel bias for unobserved channels
        self.channel_bias = nn.Parameter(torch.randn(1, self.n_channels, self.d_model))

        # Mixer blocks (from IMTS_Mixer)
        self.mixer_blocks = nn.ModuleList(
            [
                MixerBlock(self.d_model, self.n_channels, self.n_heads)
                for _ in range(self.n_layers)
            ]
        )

        # Output projection
        if self.d_model != self.d_out:
            self.out_proj = nn.Linear(self.d_model, self.d_out)
        else:
            self.out_proj = nn.Identity()

        # Time encoding for prediction
        self.time_encoder_pred = nn.Sequential(
            nn.Linear(1, self.d_time),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_time, self.d_out),
        )

        # Final decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.d_out, self.d_out),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_out, 1),
        )

        # Layer normalization and dropout for better training
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(0.1)

    def forecasting(self, tp_to_predict, observed_data, observed_tp, observed_mask):
        """
        tp_to_predict: (B, Lp) - prediction timestamps
        observed_data: (B, L, N) - observed values
        observed_tp: (B, L) - observed timestamps
        observed_mask: (B, L, N) - observation mask
        """
        B, L, N = observed_data.shape
        Lp = tp_to_predict.shape[1]

        print(f"🔍 Input shape: {observed_data.shape} (3D - using Adaptive Patching)")

        # Reshape data for processing: (B, L, N) -> (B, N, L, 1)
        x = observed_data.permute(0, 2, 1).unsqueeze(-1)  # (B, N, L, 1)
        t = observed_tp.unsqueeze(1).unsqueeze(-1).repeat(1, N, 1, 1)  # (B, N, L, 1)
        mask = observed_mask.permute(0, 2, 1).unsqueeze(-1)  # (B, N, L, 1)

        # Step 1: Adaptive patching - learn patch boundaries
        t_left, t_right = self.adaptive_patching()  # (N, n_patches)
        print(
            f"📦 Adaptive patches: {self.n_patches} patches with learnable boundaries"
        )

        # Step 2: Adaptive weighted aggregation within patches
        h_p = self.adaptive_aggregation(
            x, t, mask, t_left, t_right
        )  # (B, N, n_patches, d_model)
        print(f"✅ Adaptive aggregation: {h_p.shape}")

        # Step 3: Query-based aggregation across patches
        z = self.query_aggregation(h_p)  # (B, N, d_model)
        print(f"✅ Query aggregation: {z.shape}")

        # Step 4: Handle unobserved channels
        unobserved_mask = (
            (observed_mask.sum(dim=1) == 0).float().unsqueeze(-1)
        )  # (B, N, 1)
        z = z * (1 - unobserved_mask) + self.channel_bias * unobserved_mask
        z = z + self.channel_bias  # Add bias to all channels

        # Step 5: Apply layer normalization and dropout
        z = self.layer_norm(z)
        z = self.dropout(z)

        # Step 6: Apply mixer blocks for cross-channel and feature mixing
        for i, mixer_block in enumerate(self.mixer_blocks):
            z_prev = z
            z = mixer_block(z)
            z = self.layer_norm(z)  # Layer norm after each block
            z = self.dropout(z)
            print(f"✅ Mixer block {i + 1}: {z.shape}")

        # Step 7: Output projection
        z = self.out_proj(z)  # (B, N, d_out)

        # Step 8: Prediction generation
        z = z.unsqueeze(2).repeat(1, 1, Lp, 1)  # (B, N, Lp, d_out)

        # Encode prediction timestamps
        t_pred_enc = self.time_encoder_pred(
            tp_to_predict.unsqueeze(1).unsqueeze(-1).repeat(1, N, 1, 1)
        )  # (B, N, Lp, d_out)

        # Combine channel representations with prediction times
        decoder_input = z * t_pred_enc  # (B, N, Lp, d_out)

        # Final prediction
        output = self.decoder(decoder_input).squeeze(-1)  # (B, N, Lp)

        # Reshape for evaluation framework: (B, N, Lp) -> (1, B, Lp, N)
        output = output.permute(0, 2, 1).unsqueeze(0)  # (1, B, Lp, N)

        print(f"🎯 Final output: {output.shape}")
        return output
