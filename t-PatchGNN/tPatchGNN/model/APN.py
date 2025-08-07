import torch
import torch.nn as nn
import torch.nn.functional as F


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

        # Each series learns its own patch boundaries - this makes sense!
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
        # We need to compute: sum over L of (alpha * x_aug)
        # alpha: (B, N, L, P), x_aug: (B, N, L, F)
        # We want: (B, N, P, F)

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


class tAPN(nn.Module):
    def __init__(self, args):
        super(tAPN, self).__init__()
        self.n_patches = args.npatch
        self.d_model = args.hid_dim
        self.d_te = args.te_dim
        self.n_series = args.ndim
        self.t_obs = args.t_obs  # Total observation window size

        self.time_embedding = LearnableTimeEmbedding(self.d_te)
        self.adaptive_patching = AdaptivePatching(
            self.n_series, self.n_patches, self.t_obs
        )
        self.weighted_aggregation = WeightedAggregation(self.n_series)
        self.projection = nn.Linear(1 + self.d_te, self.d_model)
        self.query_aggregation = QueryBasedAggregation(self.d_model, self.n_patches)

        self.decoder = nn.Sequential(
            nn.Linear(self.d_model + self.d_te, self.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model, 1),
        )

    def forecasting(self, t_pred, x, t, mask):
        # x: (B, L, D) - observed data
        # t: (B, L) - observed timestamps
        # mask: (B, L, D) - observed mask
        # t_pred: (B, Lp) - prediction timestamps

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

        t_left, t_right = self.adaptive_patching()  # (D, P)

        h_p = self.weighted_aggregation(
            x_aug, t_expanded, t_left, t_right
        )  # (B, D, P, 1+d_te)
        h_p = self.projection(h_p)  # (B, D, P, d_model)

        h_final = self.query_aggregation(h_p)  # (B, D, d_model)

        # Decoder
        Lp = t_pred.shape[1]
        h_final_re = h_final.unsqueeze(2).repeat(1, 1, Lp, 1)  # (B, D, Lp, d_model)

        t_pred_re = (
            t_pred.unsqueeze(1).unsqueeze(-1).repeat(1, D, 1, 1)
        )  # (B, D, Lp, 1)
        te_pred = self.time_embedding(t_pred_re)  # (B, D, Lp, d_te)

        decoder_input = torch.cat([h_final_re, te_pred], dim=-1)

        output = self.decoder(decoder_input).squeeze(-1)  # (B, D, Lp)

        return output.permute(0, 2, 1).unsqueeze(0)  # (1, B, Lp, D)
