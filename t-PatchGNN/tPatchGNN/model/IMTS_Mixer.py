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


class ChannelAggregation(nn.Module):
    def __init__(self, d_model, d_time):
        super(ChannelAggregation, self).__init__()
        self.observation_encoder = ObservationEncoder(d_model, d_time)
        self.weight_net = nn.Sequential(
            nn.Linear(1, d_time), nn.ReLU(inplace=True), nn.Linear(d_time, d_model)
        )

    def forward(self, v, t, mask):
        # v: (B, N, L, 1)
        # t: (B, N, L, 1)
        # mask: (B, N, L, 1)

        h = self.observation_encoder(v, t)  # (B, N, L, d_model)

        # Compute weights
        w = self.weight_net(t)  # (B, N, L, d_model)
        w = w * mask + (1 - mask) * (-1e8)
        w = F.softmax(w, dim=2)

        # Weighted sum
        z = torch.sum(w * h, dim=2)  # (B, N, d_model)
        return z


class PatchAggregation(nn.Module):
    def __init__(self, d_model, d_time):
        super(PatchAggregation, self).__init__()
        self.observation_encoder = ObservationEncoder(d_model, d_time)
        self.weight_net = nn.Sequential(
            nn.Linear(1, d_time), nn.ReLU(inplace=True), nn.Linear(d_time, d_model)
        )

    def forward(self, v, t, mask):
        # v: (B, N, M, L, 1)
        # t: (B, N, M, L, 1)
        # mask: (B, N, M, L, 1)

        B, N, M, L, _ = v.shape

        # If there's only 1 patch, we can simplify the process
        if M == 1:
            # Squeeze out the patch dimension and use regular channel aggregation
            v_squeezed = v.squeeze(2)  # (B, N, L, 1)
            t_squeezed = t.squeeze(2)  # (B, N, L, 1)
            mask_squeezed = mask.squeeze(2)  # (B, N, L, 1)
            
            # Encode observations
            h = self.observation_encoder(v_squeezed, t_squeezed)  # (B, N, L, d_model)
            
            # Compute attention weights
            w = self.weight_net(t_squeezed)  # (B, N, L, d_model)
            w = w * mask_squeezed + (1 - mask_squeezed) * (-1e8)
            w = F.softmax(w, dim=2)  # attention across time
            
            # Weighted sum across time dimension
            z = torch.sum(w * h, dim=2)  # (B, N, d_model)
            return z

        # Reshape to process patches
        v_flat = v.reshape(B * N * M, L, 1)  # (B*N*M, L, 1)
        t_flat = t.reshape(B * N * M, L, 1)  # (B*N*M, L, 1)
        mask_flat = mask.reshape(B * N * M, L, 1)  # (B*N*M, L, 1)

        # Encode observations within each patch
        h = self.observation_encoder(v_flat, t_flat)  # (B*N*M, L, d_model)

        # Compute attention weights
        w = self.weight_net(t_flat)  # (B*N*M, L, d_model)
        w = w * mask_flat + (1 - mask_flat) * (-1e8)
        w = F.softmax(w, dim=1)  # attention across time within each patch

        # Weighted sum across time dimension
        patch_repr = torch.sum(w * h, dim=1)  # (B*N*M, d_model)

        # Reshape back to separate patches
        patch_repr = patch_repr.reshape(B, N, M, -1)  # (B, N, M, d_model)

        # Now aggregate across patches for each channel
        # Compute patch importance weights based on mask coverage
        patch_coverage = mask.sum(dim=3, keepdim=True)  # (B, N, M, 1) - number of observations per patch
        
        # Add small epsilon to avoid division by zero and ensure valid softmax
        patch_coverage = patch_coverage + 1e-8
        patch_weights = F.softmax(patch_coverage, dim=2)  # (B, N, M, 1)

        # Weighted sum across patches
        z = torch.sum(patch_weights * patch_repr, dim=2)  # (B, N, d_model)

        return z


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


class IMTS_Mixer(nn.Module):
    def __init__(self, args):
        super(IMTS_Mixer, self).__init__()
        self.n_channels = args.ndim
        self.d_model = args.hid_dim
        self.d_time = args.te_dim
        self.n_layers = args.nlayer
        self.d_out = args.d_out if hasattr(args, "d_out") else self.d_model

        # Support both patched and non-patched data
        # Always create both aggregation modules since we'll detect format at runtime
        self.patch_aggregation = PatchAggregation(self.d_model, self.d_time)
        self.channel_aggregation = ChannelAggregation(self.d_model, self.d_time)

        self.channel_bias = nn.Parameter(torch.randn(1, self.n_channels, self.d_model))

        self.mixer_blocks = nn.ModuleList(
            [MixerBlock(self.d_model, self.n_channels) for _ in range(self.n_layers)]
        )

        if self.d_model != self.d_out:
            self.out_proj = nn.Linear(self.d_model, self.d_out)
        else:
            self.out_proj = nn.Identity()

        self.decoder = nn.Sequential(
            nn.Linear(self.d_out, self.d_out),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_out, 1),
        )

        self.time_encoder_pred = nn.Sequential(
            nn.Linear(1, self.d_time),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_time, self.d_out),
        )

    def forecasting(self, tp_to_predict, observed_data, observed_tp, observed_mask):
        """
        Forecasting method compatible with both patched and non-patched data.

        For patched data:
        - observed_data: (B, M, L, N) where M is number of patches
        - observed_tp: (B, M, L, N)
        - observed_mask: (B, M, L, N)
        - tp_to_predict: (B, Lp)

        For non-patched data:
        - observed_data: (B, L, N)
        - observed_tp: (B, L, N) or (B, L)
        - observed_mask: (B, L, N)
        - tp_to_predict: (B, Lp)
        """

        # Detect data format at runtime based on dimensions
        if len(observed_data.shape) == 4:
            # Handle patched data: (B, M, L, N)
            B, M, L, N = observed_data.shape

            # Reshape for processing: (B, N, M, L, 1)
            x = observed_data.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, N, M, L, 1)
            t = observed_tp.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, N, M, L, 1)
            mask = observed_mask.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, N, M, L, 1)

            # Use patch aggregation
            z = self.patch_aggregation(x, t, mask)  # (B, N, d_model)
            is_patched = True

        else:
            # Handle non-patched data: (B, L, N)
            B, L, N = observed_data.shape

            # Reshape for processing
            x = observed_data.permute(0, 2, 1).unsqueeze(-1)  # (B, N, L, 1)

            # Handle different timestamp formats
            if len(observed_tp.shape) == 2:  # (B, L)
                t = (
                    observed_tp.unsqueeze(1).unsqueeze(-1).repeat(1, N, 1, 1)
                )  # (B, N, L, 1)
            else:  # (B, L, N)
                t = observed_tp.permute(0, 2, 1).unsqueeze(-1)  # (B, N, L, 1)

            mask = observed_mask.permute(0, 2, 1).unsqueeze(-1)  # (B, N, L, 1)

            # Use channel aggregation
            z = self.channel_aggregation(x, t, mask)  # (B, N, d_model)
            is_patched = False

        # Handle unobserved channels
        if is_patched:
            # For patched data: sum over patch and time dimensions
            unobserved_mask = (observed_mask.sum(dim=(1, 2)) == 0).unsqueeze(-1)  # (B, N, 1)
        else:
            # For non-patched data: sum over time dimension  
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

        # Prepare prediction
        Lp = tp_to_predict.shape[1]
        z = z.unsqueeze(2).repeat(1, 1, Lp, 1)  # (B, N, Lp, d_out)

        t_pred_enc = self.time_encoder_pred(
            tp_to_predict.unsqueeze(1).unsqueeze(-1).repeat(1, N, 1, 1)
        )  # (B, N, Lp, d_out)

        decoder_input = z * t_pred_enc

        output = self.decoder(decoder_input).squeeze(-1)  # (B, N, Lp)

        return output.permute(0, 2, 1).unsqueeze(0)  # (1, B, Lp, N)
