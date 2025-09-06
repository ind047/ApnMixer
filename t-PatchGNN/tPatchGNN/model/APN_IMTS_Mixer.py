import torch
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
    def __init__(self, n_series, n_patches, t_obs):
        super(AdaptivePatching, self).__init__()
        self.n_series = n_series
        self.n_patches = n_patches
        self.t_obs = t_obs

        # Each series learns its own patch boundaries
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


class AdaptivePatchAggregation(nn.Module):
    """Aggregates adaptive patches using attention mechanism"""

    def __init__(self, d_model, n_patches):
        super(AdaptivePatchAggregation, self).__init__()
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
    def __init__(self, d_model, n_channels):
        super().__init__()
        self.channel_norm = RMSNorm(n_channels)  # Normalize across channels
        self.channel_mlp = nn.Sequential(
            nn.Linear(n_channels, n_channels), nn.ReLU(inplace=True)
        )
        self.hidden_norm = RMSNorm(d_model)  # Normalize across hidden features
        self.hidden_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: (B, N, D)

        # Channel mixing
        residual = x
        x = self.channel_norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.channel_mlp(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + residual

        # Hidden mixing
        residual = x
        x = self.hidden_norm(x)
        x = self.hidden_mlp(x)
        x = x + residual

        return x


class APN_IMTS_Mixer(nn.Module):
    def __init__(self, args):
        super(APN_IMTS_Mixer, self).__init__()
        self.n_patches = args.npatch
        self.d_model = args.hid_dim
        self.d_time = args.te_dim
        self.n_series = args.ndim
        self.n_layers = args.nlayer
        self.t_obs = args.t_obs  # Total observation window size
        self.d_out = args.d_out if hasattr(args, "d_out") else self.d_model

        # APN Frontend Components
        self.time_embedding = LearnableTimeEmbedding(self.d_time)
        self.adaptive_patching = AdaptivePatching(
            self.n_series, self.n_patches, self.t_obs
        )
        self.weighted_aggregation = WeightedAggregation(self.n_series)
        self.projection = nn.Linear(1 + self.d_time, self.d_model)
        self.patch_aggregation = AdaptivePatchAggregation(self.d_model, self.n_patches)

        # IMTS_Mixer Backend Components
        self.channel_bias = nn.Parameter(torch.randn(1, self.n_series, self.d_model))

        self.mixer_blocks = nn.ModuleList(
            [MixerBlock(self.d_model, self.n_series) for _ in range(self.n_layers)]
        )

        if self.d_model != self.d_out:
            self.out_proj = nn.Linear(self.d_model, self.d_out)
        else:
            self.out_proj = nn.Identity()

        self.decoder = nn.Sequential(
            nn.Linear(self.d_out + self.d_time, self.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model, 1),
        )

    def forecasting(self, tp_to_predict, observed_data, observed_tp, observed_mask):
        """
        Forecasting method that uses APN's adaptive patching as frontend to IMTS_Mixer.

        Args:
        - observed_data: (B, L, N) - observed data (non-patched format expected)
        - observed_tp: (B, L) - observed timestamps
        - observed_mask: (B, L, N) - observed mask
        - tp_to_predict: (B, Lp) - prediction timestamps
        """

        # Handle both patched and non-patched input formats
        if len(observed_data.shape) == 4:
            # If patched data comes in, flatten it back to (B, L, N)
            B, M, L, N = observed_data.shape
            observed_data = observed_data.reshape(B, M * L, N)
            observed_tp = observed_tp.reshape(B, M * L, N)[
                :, :, 0
            ]  # Take first channel's timestamps
            observed_mask = observed_mask.reshape(B, M * L, N)

        B, L, N = observed_data.shape

        # Expand time to match data dimensions: (B, L) -> (B, L, N)
        t_expanded = observed_tp.unsqueeze(-1).expand(B, L, N)

        # Reshape for APN processing: (B, L, N) -> (B, N, L, 1)
        x = observed_data.permute(0, 2, 1).unsqueeze(-1)  # (B, N, L, 1)
        t_expanded = t_expanded.permute(0, 2, 1).unsqueeze(-1)  # (B, N, L, 1)
        mask = observed_mask.permute(0, 2, 1).unsqueeze(-1)  # (B, N, L, 1)

        # === APN Frontend: Adaptive Patching and Aggregation ===

        # Apply mask to x first, then add time embedding
        x = x * mask  # Apply mask to data only
        te = self.time_embedding(t_expanded)  # (B, N, L, d_time)
        x_aug = torch.cat([x, te], dim=-1)  # (B, N, L, 1+d_time)

        # Get adaptive patch boundaries
        t_left, t_right = self.adaptive_patching()  # (N, P)

        # Weighted aggregation across adaptive patches
        h_p = self.weighted_aggregation(
            x_aug, t_expanded, t_left, t_right
        )  # (B, N, P, 1+d_time)

        # Project to model dimension
        h_p = self.projection(h_p)  # (B, N, P, d_model)

        # Aggregate patches with attention
        z = self.patch_aggregation(h_p)  # (B, N, d_model)

        # === IMTS_Mixer Backend ===

        # Handle unobserved channels
        unobserved_mask = (observed_mask.sum(dim=1) == 0).unsqueeze(-1)  # (B, N, 1)
        z = (
            z * (1 - unobserved_mask.float())
            + self.channel_bias * unobserved_mask.float()
        )
        z = z + self.channel_bias

        # Apply mixer blocks
        for mixer_block in self.mixer_blocks:
            z = mixer_block(z)

        z = self.out_proj(z)

        # === Decoder ===

        # Prepare prediction
        Lp = tp_to_predict.shape[1]
        h_final_re = z.unsqueeze(2).repeat(1, 1, Lp, 1)  # (B, N, Lp, d_model)

        t_pred_re = (
            tp_to_predict.unsqueeze(1).unsqueeze(-1).repeat(1, N, 1, 1)
        )  # (B, N, Lp, 1)
        te_pred = self.time_embedding(t_pred_re)  # (B, N, Lp, d_time)

        decoder_input = torch.cat(
            [h_final_re, te_pred], dim=-1
        )  # (B, N, Lp, d_model + d_time)

        output = self.decoder(decoder_input).squeeze(-1)  # (B, N, Lp)

        return output.permute(0, 2, 1).unsqueeze(0)  # (1, B, Lp, N)
