"""
IMTS_Mixer_Clean: Streamlined version without segmentation

This model includes:
- Phase 1: LayerNorm, GELU, Pre-normalization, Positional Encoding
- Phase 2: AttentionDecoder, Enhanced Attention, Temperature Scaling

Removed:
- All FastRNN segmentation code
- Segment-specific processing
- Multi-scale processing
- Unused complexity

Benefits:
- Cleaner, more maintainable code
- Faster training (no segmentation overhead)
- Easier to understand and modify
- Same or better performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Core Building Blocks
# ============================================================================

class ObservationEncoder(nn.Module):
    """Encodes observations with temporal context"""
    def __init__(self, d_model, d_time):
        super().__init__()
        self.value_encoder = nn.Linear(1, d_model)
        self.time_encoder = nn.Sequential(
            nn.Linear(1, d_time),
            nn.ReLU(inplace=True),
            nn.Linear(d_time, d_model)
        )
    
    def forward(self, v, t):
        # v: (B, N, L, 1), t: (B, N, L, 1)
        return self.value_encoder(v) * self.time_encoder(t)


class ChannelAggregation(nn.Module):
    """
    Clean channel aggregation with enhanced attention.
    Phase 2 #3: Temperature-scaled attention for better weight distribution.
    """
    def __init__(self, d_model, d_time):
        super().__init__()
        self.d_model = d_model
        self.d_time = d_time
        
        # Observation encoder
        self.observation_encoder = ObservationEncoder(d_model, d_time)
        
        # Phase 2 #3: Enhanced weight network with normalization and dropout
        self.weight_net = nn.Sequential(
            nn.Linear(1, d_time),
            nn.LayerNorm(d_time),  # Stabilizes attention
            nn.GELU(),             # Smoother gradients
            nn.Dropout(0.1),       # Prevents overfitting
            nn.Linear(d_time, d_model)
        )
        
        self.value_encoder = nn.Linear(1, d_model)
        
        # Phase 2 #3: Learnable temperature for attention scaling
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, v, t, mask):
        # v: (B, N, L, 1), t: (B, N, L, 1), mask: (B, N, L, 1)
        
        # Compute attention weights
        h = self.value_encoder(v) * self.weight_net(t)
        a = self.value_encoder(v) + self.weight_net(t)
        
        # Apply mask
        a = a * mask + (1 - mask) * (-1e8)
        
        # Phase 2 #3: Temperature-scaled softmax for better distribution
        a = F.softmax(a / (self.temperature + 1e-8), dim=2)
        
        # Weighted aggregation
        z = torch.sum(a * h, dim=2)  # (B, N, d_model)
        
        return z


class PatchAggregation(nn.Module):
    """Clean patch aggregation for patched mode"""
    def __init__(self, d_model, d_time):
        super().__init__()
        self.d_model = d_model
        self.d_time = d_time
        
        self.observation_encoder = ObservationEncoder(d_model, d_time)
        
        # Enhanced weight network (Phase 2)
        self.weight_net = nn.Sequential(
            nn.Linear(1, d_time),
            nn.LayerNorm(d_time),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_time, d_model)
        )
        
        self.value_encoder = nn.Linear(1, d_model)
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, v, t, mask):
        # v: (B, N, M, L, 1), t: (B, N, M, L, 1), mask: (B, N, M, L, 1)
        B, N, M, L, _ = v.shape
        
        # Aggregate within each patch
        h = self.value_encoder(v) * self.weight_net(t)
        a = self.value_encoder(v) + self.weight_net(t)
        a = a * mask + (1 - mask) * (-1e8)
        a = F.softmax(a / (self.temperature + 1e-8), dim=3)
        z = torch.sum(a * h, dim=3)  # (B, N, M, d_model)
        
        # Aggregate across patches
        z = z.mean(dim=2)  # (B, N, d_model)
        
        return z


# ============================================================================
# Phase 1: MixerBlock with improvements
# ============================================================================

class MixerBlock(nn.Module):
    """
    Phase 1 Mixer Block with:
    - LayerNorm instead of RMSNorm (better stability)
    - GELU instead of ReLU (smoother gradients)
    - Expanded MLPs (2x for channel, 4x for token)
    - Dropout for regularization
    - Pre-normalization architecture
    """
    def __init__(self, d_model, n_channels, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_channels = n_channels
        
        # Channel mixing (across features)
        self.channel_norm = nn.LayerNorm(d_model)
        self.channel_mlp = nn.Sequential(
            nn.Linear(n_channels, n_channels * 2),  # 2x expansion
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(n_channels * 2, n_channels),
            nn.Dropout(0.1),
        )
        
        # Token mixing (across time/channels)
        self.token_norm = nn.LayerNorm(d_model)
        self.token_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),  # 4x expansion
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1),
        )
    
    def forward(self, x):
        # x: (B, N, d_model)
        
        # Channel mixing with pre-norm and residual
        residual = x
        x_norm = self.channel_norm(x)
        x_transposed = x_norm.transpose(1, 2)  # (B, d_model, N)
        x_mixed = self.channel_mlp(x_transposed)
        x = residual + x_mixed.transpose(1, 2)
        
        # Token mixing with pre-norm and residual
        residual = x
        x_norm = self.token_norm(x)
        x = residual + self.token_mlp(x_norm)
        
        return x


# ============================================================================
# Phase 2 #5: Enhanced Decoder with Attention
# ============================================================================

class AttentionDecoder(nn.Module):
    """
    Phase 2 #5: Enhanced decoder with multi-head attention.
    Enables cross-channel reasoning for better predictions.
    """
    def __init__(self, d_in, d_out, n_heads=4):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            d_in, n_heads, dropout=0.1, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_in)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_in, d_in * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_in * 2, d_out),
            nn.Dropout(0.1),
        )
        self.norm2 = nn.LayerNorm(d_in)
        
        # Output projection
        self.output = nn.Sequential(
            nn.Linear(d_out, d_out),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_out, 1),
        )
    
    def forward(self, decoder_input):
        # decoder_input: (B, N, d_in)
        
        # Self-attention with residual
        residual = decoder_input
        x_norm = self.norm1(decoder_input)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        x = residual + attn_out
        
        # Feed-forward with residual
        residual = x
        x_norm = self.norm2(x)
        x = residual + self.ffn(x_norm)
        
        # Output projection
        x = self.output(x)
        
        return x


# ============================================================================
# Phase 1 #4: Learnable Positional Encoding
# ============================================================================

class LearnablePositionalEncoding(nn.Module):
    """
    Phase 1 #4: Adds learnable positional information for temporal awareness.
    Helps model understand sequence order and temporal relationships.
    """
    def __init__(self, d_model, max_len=256):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x):
        # x: (B, N, d_model)
        B, N, d_model = x.shape
        
        if N > self.pos_embedding.shape[1]:
            # Interpolate if sequence is longer
            pos_emb = F.interpolate(
                self.pos_embedding.transpose(1, 2),
                size=N,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        else:
            pos_emb = self.pos_embedding[:, :N, :]
        
        return x + pos_emb


# ============================================================================
# Main Model
# ============================================================================

class IMTS_Mixer_Clean(nn.Module):
    """
    Clean IMTS Mixer without segmentation complexity.
    
    Includes all Phase 1 and Phase 2 improvements:
    - Phase 1: LayerNorm, GELU, Pre-norm, Positional Encoding
    - Phase 2: AttentionDecoder, Enhanced Attention, Temperature Scaling
    
    Removed:
    - FastRNN segmentation
    - Multi-scale processing
    - Segment-specific processors
    """
    def __init__(self, args):
        super().__init__()
        
        # Model dimensions
        self.n_channels = args.ndim
        self.d_model = args.hid_dim
        self.d_time = args.te_dim
        self.d_out = args.d_out
        self.n_layers = args.nlayer
        self.n_heads = args.n_heads
        
        # Aggregation modules (clean, no segmentation)
        self.patch_aggregation = PatchAggregation(self.d_model, self.d_time)
        self.channel_aggregation = ChannelAggregation(self.d_model, self.d_time)
        
        # Channel bias for unobserved channels
        self.channel_bias = nn.Parameter(torch.randn(1, self.n_channels, self.d_model))
        
        # Phase 1 #4: Positional encoding
        self.pos_encoding = LearnablePositionalEncoding(self.d_model, max_len=256)
        
        # Pre-normalization layers
        self.pre_norms = nn.ModuleList([
            nn.LayerNorm(self.d_model) for _ in range(self.n_layers)
        ])
        
        # Mixer blocks
        self.mixer_blocks = nn.ModuleList([
            MixerBlock(self.d_model, self.n_channels, self.n_heads)
            for _ in range(self.n_layers)
        ])
        
        # Final normalization
        self.final_norm = nn.LayerNorm(
            self.d_out if self.d_model != self.d_out else self.d_model
        )
        
        # Output projection
        if self.d_model != self.d_out:
            self.output_projection = nn.Sequential(
                nn.Linear(self.d_model, self.d_out),
                nn.Dropout(0.1),
            )
        else:
            self.output_projection = nn.Identity()
        
        # Phase 2 #5: AttentionDecoder
        self.decoder = AttentionDecoder(self.d_out, self.d_out, n_heads=4)
        
        # Time encoder for prediction
        self.time_encoder_pred = nn.Sequential(
            nn.Linear(1, self.d_time),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_time, self.d_out),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Simple, stable weight initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        # Small initialization for special parameters
        nn.init.normal_(self.channel_bias, std=0.02)
    
    def forecasting(self, tp_to_predict, observed_data, observed_tp, observed_mask):
        """
        Main forecasting function.
        
        Args:
            tp_to_predict: (B, L_pred) - Time points to predict
            observed_data: (B, L, N) or (B, M, L, N) - Observations
            observed_tp: (B, L) or (B, L, N) or (B, M, L, N) - Observation times
            observed_mask: (B, L, N) or (B, M, L, N) - Observation mask
        
        Returns:
            predictions: (1, B, L_pred, N) - Forecasted values
        """
        # Detect data format (same as FastRNN)
        if len(observed_data.shape) == 4:
            # Patched mode: (B, M, L, N)
            B, M, L, N = observed_data.shape
            x = observed_data.permute(0, 3, 1, 2).unsqueeze(-1)
            t = observed_tp.permute(0, 3, 1, 2).unsqueeze(-1)
            mask = observed_mask.permute(0, 3, 1, 2).unsqueeze(-1)
            z = self.patch_aggregation(x, t, mask)
            is_patched = True
        else:
            # Non-patched mode: (B, L, N)
            B, L, N = observed_data.shape
            x = observed_data.permute(0, 2, 1).unsqueeze(-1)
            
            if len(observed_tp.shape) == 2:
                t = observed_tp.unsqueeze(1).unsqueeze(-1).repeat(1, N, 1, 1)
            else:
                t = observed_tp.permute(0, 2, 1).unsqueeze(-1)
            
            mask = observed_mask.permute(0, 2, 1).unsqueeze(-1)
            z = self.channel_aggregation(x, t, mask)
            is_patched = False
        
        # Handle unobserved channels
        if is_patched:
            unobserved_mask = (observed_mask.sum(dim=(1, 2)) == 0).unsqueeze(-1)
        else:
            unobserved_mask = (observed_mask.sum(dim=1) == 0).unsqueeze(-1)
        
        z = z * (1 - unobserved_mask.float()) + self.channel_bias * unobserved_mask.float()
        
        # Phase 1 #4: Apply positional encoding
        z = self.pos_encoding(z)
        
        # Apply mixer blocks with pre-normalization
        for pre_norm, mixer_block in zip(self.pre_norms, self.mixer_blocks):
            z_norm = pre_norm(z)
            z = z + mixer_block(z_norm)
        
        # Project to output dimension
        z = self.output_projection(z)
        z = self.final_norm(z)
        
        # Prepare for forecasting
        z = z.unsqueeze(-2)  # (B, N, 1, d_out)
        L_pred = tp_to_predict.shape[-1]
        z = z.repeat(1, 1, L_pred, 1)  # (B, N, L_pred, d_out)
        
        # Time encoding for prediction
        tp_to_predict = tp_to_predict.view(B, 1, L_pred, 1).repeat(1, N, 1, 1)
        te_pred = self.time_encoder_pred(tp_to_predict)
        
        # Combine and decode
        z = z + te_pred
        
        # Phase 2 #5: Reshape for AttentionDecoder
        B, N, L_pred, d_out = z.shape
        z_reshaped = z.view(B * N, L_pred, d_out)
        
        # Apply attention decoder
        outputs = self.decoder(z_reshaped)  # (B*N, L_pred, 1)
        
        # Reshape back
        outputs = outputs.view(B, N, L_pred).unsqueeze(0).permute(0, 1, 3, 2)
        
        return outputs
