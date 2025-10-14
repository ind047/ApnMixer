import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming APN.py is in the same directory
from .APN import (
    LearnableTimeEmbedding,
    AdaptivePatching,
    WeightedAggregation,
)


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

        # Projection to match dimensions for the mixer
        self.projection = nn.Linear(1 + self.d_te, self.d_model)

        # --- MLP-Mixer Style Backend (simplified) ---
        # Instead of using full PatchTSMixer, use custom MLP blocks
        self.mixer_layers = nn.ModuleList(
            [self._create_mixer_block() for _ in range(args.nlayer)]
        )
        self.norm = nn.LayerNorm(self.d_model)

        # --- Simple aggregation and Decoder ---
        # Simple aggregation: average across patches using mean
        # self.patch_aggregation = nn.AdaptiveAvgPool1d(1)  # Pool across patch dimension

        self.decoder = nn.Sequential(
            nn.Linear(self.d_model + self.d_te, self.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model, 1),
        )

    def _create_mixer_block(self):
        """Create a simplified MLP-Mixer block"""
        return nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.Dropout(0.1),
        )

    def forecasting(self, t_pred, x, t, mask):
        # Handle different input shapes
        # x might be (n_traj_samples, B, L, D) or (B, L, D)
        # t might be (n_traj_samples, B, L) or (B, L)
        # mask might be (n_traj_samples, B, L, D) or (B, L, D)
        # t_pred might be (n_traj_samples, B, Lp) or (B, Lp)

        # Handle different input shapes
        # Data comes as (B, n_traj_samples=1, L, D) or (B, L, D)
        if len(x.shape) == 4 and x.shape[1] == 1:
            x = x.squeeze(1)  # (B, 1, L, D) -> (B, L, D)
        if len(t.shape) == 4 and t.shape[1] == 1:
            t = t.squeeze(1)  # (B, 1, L, D) -> (B, L, D)
            # t should be (B, L), so we need to take just one channel
            t = t[:, :, 0]  # (B, L, D) -> (B, L)
        if len(mask.shape) == 4 and mask.shape[1] == 1:
            mask = mask.squeeze(1)  # (B, 1, L, D) -> (B, L, D)
        if len(t_pred.shape) == 3 and t_pred.shape[1] == 1:
            t_pred = t_pred.squeeze(1)  # (B, 1, Lp) -> (B, Lp)

        B, L, D = x.shape

        # Reshape for APN processing: (B, L, D) -> (B, D, L, 1)
        x_reshaped = x.permute(0, 2, 1).unsqueeze(-1)
        t_expanded = t.unsqueeze(-1).expand(B, L, D).permute(0, 2, 1).unsqueeze(-1)
        mask_reshaped = mask.permute(0, 2, 1).unsqueeze(-1)

        # Apply mask and create time embeddings
        x_masked = x_reshaped * mask_reshaped
        te = self.time_embedding(t_expanded)
        x_aug = torch.cat([x_masked, te], dim=-1)

        # 1. Adaptive Patching and Weighted Aggregation
        t_left, t_right = self.adaptive_patching()
        h_p = self.weighted_aggregation(
            x_aug, t_expanded, t_left, t_right
        )  # (B, D, P, 1+d_te)

        # 2. Project to d_model for the mixer
        h_p_proj = self.projection(h_p)  # (B, D, P, d_model)

        # 3. Apply MLP-Mixer style processing
        # h_p_proj shape: (B, D, P, d_model)

        # Apply mixer layers with residual connections
        mixer_output = h_p_proj
        for mixer_layer in self.mixer_layers:
            # Reshape for mixer: (B, D, P, d_model) -> (B*D, P, d_model)
            B_curr, D_curr, P_curr, d_curr = mixer_output.shape
            mixer_input = mixer_output.reshape(B_curr * D_curr, P_curr, d_curr)

            # Apply mixer block
            mixer_processed = mixer_layer(mixer_input)

            # Reshape back and add residual connection
            mixer_reshaped = mixer_processed.reshape(B_curr, D_curr, P_curr, d_curr)
            mixer_output = mixer_reshaped + mixer_output  # Residual connection

        # Apply final normalization
        mixer_output_final = self.norm(mixer_output)

        # 4. Aggregate patch representations using simple mean
        # mixer_output_final: (B, D, P, d_model)
        # We want to aggregate across the patch dimension P
        h_final = mixer_output_final.mean(dim=2)  # (B, D, d_model)

        # 5. Decode for forecasting
        Lp = t_pred.shape[1]
        h_final_re = h_final.unsqueeze(2).repeat(1, 1, Lp, 1)

        t_pred_re = t_pred.unsqueeze(1).unsqueeze(-1).repeat(1, D, 1, 1)
        te_pred = self.time_embedding(t_pred_re)

        decoder_input = torch.cat([h_final_re, te_pred], dim=-1)
        output = self.decoder(decoder_input).squeeze(-1)  # (B, D, Lp)

        return output.permute(0, 2, 1).unsqueeze(0)  # (1, B, Lp, D)
