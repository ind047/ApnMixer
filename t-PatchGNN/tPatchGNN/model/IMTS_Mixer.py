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
        self.value_encoder = nn.Linear(1, d_model)

    def forward(self, v, t, mask):
        # v: (B, N, L, 1)
        # t: (B, N, L, 1)
        # mask: (B, N, L, 1)

        # h = self.observation_encoder(v, t)  # (B, N, L, d_model)
        h = self.value_encoder(v) * self.weight_net(t)  # (B, N, L, d_model)
        # Compute weights
        w = self.weight_net(t)  # (B, N, L, d_model)
        a = self.value_encoder(v) + w
        a = a * mask + (1 - mask) * (-1e8)
        a = F.softmax(a, dim=2)
        z = torch.sum(a * h, dim=2)  # (B, N, d_model)

        # # --- IGNORE ---
        # w = w * mask + (1 - mask) * (-1e8)
        # w = F.softmax(w, dim=2)

        # # Weighted sum
        # z = torch.sum(w * h, dim=2)  # (B, N, d_model)
        return z


class PatchAggregation(nn.Module):
    def __init__(self, d_model, d_time):
        super(PatchAggregation, self).__init__()
        self.observation_encoder = ObservationEncoder(d_model, d_time)
        self.weight_net = nn.Sequential(
            nn.Linear(1, d_time), nn.ReLU(inplace=True), nn.Linear(d_time, d_model)
        )
        self.value_encoder = nn.Linear(1, d_model)

    def forward(self, v, t, mask):
        B, N, M, L, _ = v.shape
        # print(f"🔧 PatchAggregation: {M} patches detected")

        if M == 1:
            # print(f"🚀 Using SIMPLIFIED path (single patch)")
            # Simplified path for single patch
            v_flat = v.squeeze(2)  # (B, N, L, 1)
            t_flat = t.squeeze(2)  # (B, N, L, 1)
            mask_flat = mask.squeeze(2)  # (B, N, L, 1)

            # Process like ChannelAggregation
            h = self.observation_encoder(v_flat, t_flat)  # (B, N, L, d_model)
            w = self.weight_net(t_flat)  # (B, N, L, 1)
            w = w * mask_flat + (1 - mask_flat) * (-1e8)
            w = F.softmax(w, dim=2)  # (B, N, L, 1)

            z = torch.sum(w * h, dim=2)  # (B, N, d_model)

        else:
            # print(f"🏗️ Using FULL PATCH path ({M} patches)")
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
            patch_coverage = torch.sum(mask, dim=3)  # (B, N, M, 1)
            patch_coverage = patch_coverage + 1e-8  # avoid zero division
            patch_weights = F.softmax(patch_coverage, dim=2)  # (B, N, M, 1)
            # Final aggregation across patches
            z = torch.sum(patch_weights * patch_repr, dim=2)  # (B, N, d_model)

            # Step 1: Within-patch aggregation
            # patch_representations = []
            # patch_coverage = []

            # for m in range(M):
            #     # Extract patch m from all batches and channels
            #     v_patch = v[:, :, m, :, :]  # (B, N, L, 1)
            #     t_patch = t[:, :, m, :, :]  # (B, N, L, 1)
            #     mask_patch = mask[:, :, m, :, :]  # (B, N, L, 1)

            #     # Process this patch
            #     h_patch = self.observation_encoder(
            #         v_patch, t_patch
            #     )  # (B, N, L, d_model)
            #     w_patch = self.weight_net(t_patch)  # (B, N, L, 1)
            #     w_patch = w_patch * mask_patch + (1 - mask_patch) * (-1e8)
            #     w_patch = F.softmax(w_patch, dim=2)  # (B, N, L, 1)

            #     # Aggregate within patch
            #     patch_repr = torch.sum(w_patch * h_patch, dim=2)  # (B, N, d_model)
            #     patch_representations.append(patch_repr)

            #     # Calculate patch coverage (how much data this patch has)
            #     coverage = torch.sum(mask_patch, dim=2)  # (B, N, 1)
            #     patch_coverage.append(coverage)

            # # Step 2: Stack patch representations
            # patch_repr = torch.stack(patch_representations, dim=2)  # (B, N, M, d_model)
            # patch_coverage = torch.stack(patch_coverage, dim=2)  # (B, N, M, 1)

            # # Step 3: Cross-patch aggregation
            # # Weight patches by how much data they contain
            # patch_weights = F.softmax(patch_coverage, dim=2)  # (B, N, M, 1)

            # # Debug prints
            # # print(f"patch_weights shape: {patch_weights.shape}")
            # # print(f"patch_repr shape: {patch_repr.shape}")

            # # Final aggregation across patches
            # z = torch.sum(patch_weights * patch_repr, dim=2)  # (B, N, d_model)

        return z


class MixerBlock(nn.Module):
    def __init__(self, d_model, n_channels, n_heads=4):
        super().__init__()
        # Multi-head attention for channel mixing (NEW)
        self.channel_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True
        )
        self.channel_norm = RMSNorm(n_channels)  # Normalize across features

        # OLD approach (commented out but preserved)
        self.channel_norm_old = RMSNorm(n_channels)  # Normalize across channels
        self.channel_mlp = nn.Sequential(
            nn.Linear(n_channels, n_channels), nn.ReLU(inplace=True)
        )

        # Keep the original hidden mixing
        self.hidden_norm = RMSNorm(d_model)  # Normalize across hidden features
        self.hidden_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: (B, N, D) - batch, channels, features

        # Channel mixing using multi-head attention (NEW)
        residual = x
        # x_norm = self.channel_norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        # # Apply multi-head attention across channels
        # # Each channel attends to all other channels
        # attn_output, _ = self.channel_attention(
        #     query=x_norm,  # (B, N, D)
        #     key=x_norm,  # (B, N, D)
        #     value=x_norm,  # (B, N, D)
        # )
        # x = attn_output + residual  # Residual connection

        # OLD channel mixing approach (commented out but preserved)
        #residual = x
        x = self.channel_norm_old(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.channel_mlp(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + residual

        # Hidden mixing (same as before)
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
        self.n_heads = getattr(args, "n_heads", 4)  # Default 4 heads

        # Support both patched and non-patched data
        # Always create both aggregation modules since we'll detect format at runtime
        self.patch_aggregation = PatchAggregation(self.d_model, self.d_time)
        self.channel_aggregation = ChannelAggregation(self.d_model, self.d_time)

        self.channel_bias = nn.Parameter(torch.randn(1, self.n_channels, self.d_model))

        self.mixer_blocks = nn.ModuleList(
            [
                MixerBlock(self.d_model, self.n_channels, self.n_heads)
                for _ in range(self.n_layers)
            ]
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
            unobserved_mask = (observed_mask.sum(dim=(1, 2)) == 0).unsqueeze(
                -1
            )  # (B, N, 1)
        else:
            # For non-patched data: sum over time dimension
            unobserved_mask = (observed_mask.sum(dim=1) == 0).unsqueeze(-1)  # (B, N, 1)

        # Apply channel bias only to unobserved channels (corrected implementation)
        z = (
            z * (1 - unobserved_mask.float())
            + self.channel_bias * unobserved_mask.float()
        )

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
