"""
IMTS_Mixer with FastRNN Segmentation
Extends the original IMTS_Mixer with efficient FastRNN-based segmentation for real-time temporal pattern detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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


# ============================================================================
# FastRNN Components
# ============================================================================

class FastRNNSegmenter(nn.Module):
    """Fast RNN for efficient change point detection"""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Fast projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GRU for sequential processing (faster than LSTM)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers,
                         batch_first=True, bidirectional=True)
        
        # Change point detection head
        self.change_detector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Segment state classification
        self.state_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5)  # 5 states
        )
        
    def forward(self, x, mask=None):
        # x: (B, L, D)
        B, L, D = x.shape
        
        # Project input
        x_proj = self.input_proj(x)
        
        # Apply mask if provided
        if mask is not None:
            x_proj = x_proj * mask
        
        # RNN processing
        rnn_out, _ = self.rnn(x_proj)  # (B, L, hidden_dim*2)
        
        # Change point detection
        change_probs = self.change_detector(rnn_out).squeeze(-1)  # (B, L)
        
        # State classification
        state_logits = self.state_classifier(rnn_out)  # (B, L, 5)
        
        return change_probs, state_logits
    
    def detect_segments(self, x, mask=None, threshold=0.5):
        """Extract segment boundaries from change points"""
        change_probs, state_logits = self.forward(x, mask)
        
        segments_list = []
        segment_types_list = []
        
        for b in range(x.shape[0]):
            # Find change points
            changes = torch.where(change_probs[b] > threshold)[0].cpu().numpy()
            
            # Convert to segments
            if len(changes) == 0:
                segments = [(0, x.shape[1])]
            else:
                segments = []
                start = 0
                for cp in changes:
                    if cp > start:
                        segments.append((start, int(cp)))
                        start = int(cp)
                if start < x.shape[1]:
                    segments.append((start, x.shape[1]))
            
            # Get segment types
            types = []
            for start, end in segments:
                segment_logit = state_logits[b, start:end].mean(dim=0)
                seg_type = torch.argmax(segment_logit).item()
                types.append(seg_type)
            
            segments_list.append(segments)
            segment_types_list.append(types)
        
        return segments_list, segment_types_list


class MultiScaleFastRNN(nn.Module):
    """Multi-scale FastRNN for capturing patterns at different temporal resolutions"""
    def __init__(self, input_dim, scales=[1, 2, 4], hidden_dim=32):
        super().__init__()
        self.scales = scales
        self.hidden_dim = hidden_dim
        
        # Different RNNs for different scales
        self.scale_rnns = nn.ModuleDict({
            f'scale_{s}': FastRNNSegmenter(input_dim, hidden_dim=hidden_dim, num_layers=2)
            for s in scales
        })
        
        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(len(scales) * 2, hidden_dim),  # *2 for change_probs and states
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # Output: change_prob, state
        )
        
        self.change_output = nn.Linear(2, 1)
        self.state_output = nn.Linear(2, 5)
    
    def forward(self, x, mask=None):
        scale_outputs = []
        
        for scale in self.scales:
            # Downsample for this scale
            if scale > 1:
                # Use average pooling for downsampling
                x_scaled = F.avg_pool1d(
                    x.transpose(1, 2),
                    kernel_size=scale,
                    stride=scale,
                    count_include_pad=False
                ).transpose(1, 2)
                
                if mask is not None:
                    mask_scaled = F.avg_pool1d(
                        mask.transpose(1, 2),
                        kernel_size=scale,
                        stride=scale
                    ).transpose(1, 2) > 0.5
                else:
                    mask_scaled = None
            else:
                x_scaled, mask_scaled = x, mask
            
            # Process with scale-specific RNN
            change_probs, state_logits = self.scale_rnns[f'scale_{scale}'](x_scaled, mask_scaled)
            
            # Upsample back to original resolution
            if scale > 1:
                change_probs = F.interpolate(
                    change_probs.unsqueeze(1),
                    size=x.shape[1],
                    mode='linear',
                    align_corners=False
                ).squeeze(1)
                
                state_logits = F.interpolate(
                    state_logits.transpose(1, 2),
                    size=x.shape[1],
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            
            # Take max state probability
            state_probs = F.softmax(state_logits, dim=-1).max(dim=-1)[0]
            
            scale_outputs.append(torch.stack([change_probs, state_probs], dim=-1))
        
        # Fuse multi-scale outputs
        stacked = torch.cat(scale_outputs, dim=-1)  # (B, L, n_scales*2)
        fused = self.fusion(stacked)  # (B, L, 2)
        
        # Split outputs
        change_probs = torch.sigmoid(self.change_output(fused)).squeeze(-1)  # (B, L)
        state_logits = self.state_output(fused)  # (B, L, 5)
        
        return change_probs, state_logits
    
    def detect_segments(self, x, mask=None, threshold=0.5):
        """Detect segments using multi-scale analysis"""
        change_probs, state_logits = self.forward(x, mask)
        
        segments_list = []
        segment_types_list = []
        
        for b in range(x.shape[0]):
            changes = torch.where(change_probs[b] > threshold)[0].cpu().numpy()
            
            if len(changes) == 0:
                segments = [(0, x.shape[1])]
            else:
                segments = []
                start = 0
                for cp in changes:
                    if cp > start:
                        segments.append((start, int(cp)))
                        start = int(cp)
                if start < x.shape[1]:
                    segments.append((start, x.shape[1]))
            
            types = []
            for start, end in segments:
                segment_logit = state_logits[b, start:end].mean(dim=0)
                seg_type = torch.argmax(segment_logit).item()
                types.append(seg_type)
            
            segments_list.append(segments)
            segment_types_list.append(types)
        
        return segments_list, segment_types_list


# ============================================================================
# FastRNN-based Aggregation Modules
# ============================================================================

class FastRNNChannelAggregation(nn.Module):
    """Channel aggregation with FastRNN segmentation"""
    def __init__(self, d_model, d_time, use_segmentation=True, use_multiscale=True):
        super().__init__()
        self.d_model = d_model
        self.d_time = d_time
        self.use_segmentation = use_segmentation
        
        # FastRNN segmenter
        if use_segmentation:
            if use_multiscale:
                self.segmenter = MultiScaleFastRNN(
                    input_dim=1, scales=[1, 2, 4], hidden_dim=32
                )
            else:
                self.segmenter = FastRNNSegmenter(
                    input_dim=1, hidden_dim=64, num_layers=2
                )
        
        # Observation encoder
        self.observation_encoder = ObservationEncoder(d_model, d_time)
        
        # Phase 2 #3: Enhanced weight network with normalization and dropout
        self.weight_net = nn.Sequential(
            nn.Linear(1, d_time),
            nn.LayerNorm(d_time),  # ADD NORMALIZATION
            nn.GELU(),  # CHANGE FROM RELU
            nn.Dropout(0.1),  # ADD DROPOUT
            nn.Linear(d_time, d_model)
        )
        self.value_encoder = nn.Linear(1, d_model)
        
        # Phase 2 #3: Temperature parameter for attention
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
        # Segment-specific processors
        if use_segmentation:
            self.segment_processors = nn.ModuleDict({
                '0': nn.Linear(d_model, d_model),  # Stable
                '1': nn.Linear(d_model, d_model),  # Transitional
                '2': nn.Linear(d_model, d_model),  # Critical
                '3': nn.Linear(d_model, d_model),  # Recovery
                '4': nn.Linear(d_model, d_model),  # Anomalous
            })
    
    def forward(self, v, t, mask):
        # v: (B, N, L, 1)
        B, N, L, _ = v.shape
        
        if self.use_segmentation:
            channel_representations = []
            
            for n in range(N):
                channel_v = v[:, n]  # (B, L, 1)
                channel_t = t[:, n]
                channel_mask = mask[:, n]
                
                # Get segments from FastRNN
                segments_list, segment_types_list = self.segmenter.detect_segments(
                    channel_v, channel_mask
                )
                
                # Process each batch
                batch_reprs = []
                for b in range(B):
                    segments = segments_list[b]
                    seg_types = segment_types_list[b]
                    
                    segment_reprs = []
                    segment_weights = []
                    
                    for (start, end), seg_type in zip(segments, seg_types):
                        if end <= start:
                            continue
                        
                        # Extract segment
                        v_seg = channel_v[b:b+1, start:end]
                        t_seg = channel_t[b:b+1, start:end]
                        mask_seg = channel_mask[b:b+1, start:end]
                        
                        # Encode segment
                        h = self.value_encoder(v_seg) * self.weight_net(t_seg)
                        a = self.value_encoder(v_seg) + self.weight_net(t_seg)
                        a = a * mask_seg + (1 - mask_seg) * (-1e8)
                        a = F.softmax(a, dim=1)
                        
                        # Aggregate
                        seg_repr = torch.sum(a * h, dim=1)
                        
                        # Apply segment-specific processing
                        seg_repr = self.segment_processors[str(seg_type)](seg_repr)
                        
                        segment_reprs.append(seg_repr)
                        segment_weights.append(end - start)
                    
                    if segment_reprs:
                        weights = torch.tensor(segment_weights, device=v.device, dtype=torch.float)
                        weights = F.softmax(weights, dim=0)
                        stacked = torch.stack(segment_reprs).squeeze(1)
                        batch_repr = torch.sum(weights.unsqueeze(-1) * stacked, dim=0)
                    else:
                        batch_repr = torch.zeros(self.d_model, device=v.device)
                    
                    batch_reprs.append(batch_repr)
                
                channel_repr = torch.stack(batch_reprs)
                channel_representations.append(channel_repr)
            
            z = torch.stack(channel_representations, dim=1)
        
        else:
            # Phase 2 #3: Enhanced aggregation with temperature-scaled attention
            h = self.value_encoder(v) * self.weight_net(t)
            a = self.value_encoder(v) + self.weight_net(t)
            a = a * mask + (1 - mask) * (-1e8)
            # Temperature-scaled attention for better weight distribution
            a = F.softmax(a / (self.temperature + 1e-8), dim=2)
            z = torch.sum(a * h, dim=2)
        
        return z


class FastRNNPatchAggregation(nn.Module):
    """Patch aggregation with FastRNN segmentation"""
    def __init__(self, d_model, d_time, use_segmentation=True, use_multiscale=True):
        super().__init__()
        self.d_model = d_model
        self.use_segmentation = use_segmentation
        
        # FastRNN segmenter
        if use_segmentation:
            if use_multiscale:
                self.segmenter = MultiScaleFastRNN(
                    input_dim=1, scales=[1, 2, 4], hidden_dim=32
                )
            else:
                self.segmenter = FastRNNSegmenter(
                    input_dim=1, hidden_dim=64, num_layers=2
                )
        
        # Observation encoder
        self.observation_encoder = ObservationEncoder(d_model, d_time)
        self.weight_net = nn.Sequential(
            nn.Linear(1, d_time), nn.ReLU(inplace=True), nn.Linear(d_time, d_model)
        )
        self.value_encoder = nn.Linear(1, d_model)
        
        # Segment-specific processors
        if use_segmentation:
            self.segment_processors = nn.ModuleDict({
                '0': nn.Linear(d_model, d_model),
                '1': nn.Linear(d_model, d_model),
                '2': nn.Linear(d_model, d_model),
                '3': nn.Linear(d_model, d_model),
                '4': nn.Linear(d_model, d_model),
            })
    
    def forward(self, v, t, mask):
        # v: (B, N, M, L, 1)
        B, N, M, L, _ = v.shape
        
        if self.use_segmentation:
            channel_representations = []
            
            for n in range(N):
                # Flatten patches
                channel_v = v[:, n].reshape(B, M * L, 1)
                channel_t = t[:, n].reshape(B, M * L, 1)
                channel_mask = mask[:, n].reshape(B, M * L, 1)
                
                # Get segments
                segments_list, segment_types_list = self.segmenter.detect_segments(
                    channel_v, channel_mask
                )
                
                # Process segments
                batch_reprs = []
                for b in range(B):
                    segments = segments_list[b]
                    seg_types = segment_types_list[b]
                    
                    segment_reprs = []
                    segment_weights = []
                    
                    for (start, end), seg_type in zip(segments, seg_types):
                        if end <= start:
                            continue
                        
                        v_seg = channel_v[b:b+1, start:end]
                        t_seg = channel_t[b:b+1, start:end]
                        mask_seg = channel_mask[b:b+1, start:end]
                        
                        h = self.value_encoder(v_seg) * self.weight_net(t_seg)
                        a = self.value_encoder(v_seg) + self.weight_net(t_seg)
                        a = a * mask_seg + (1 - mask_seg) * (-1e8)
                        a = F.softmax(a, dim=1)
                        
                        seg_repr = torch.sum(a * h, dim=1)
                        seg_repr = self.segment_processors[str(seg_type)](seg_repr)
                        
                        segment_reprs.append(seg_repr)
                        segment_weights.append(end - start)
                    
                    if segment_reprs:
                        weights = torch.tensor(segment_weights, device=v.device, dtype=torch.float)
                        weights = F.softmax(weights, dim=0)
                        stacked = torch.stack(segment_reprs).squeeze(1)
                        batch_repr = torch.sum(weights.unsqueeze(-1) * stacked, dim=0)
                    else:
                        batch_repr = torch.zeros(self.d_model, device=v.device)
                    
                    batch_reprs.append(batch_repr)
                
                channel_repr = torch.stack(batch_reprs)
                channel_representations.append(channel_repr)
            
            z = torch.stack(channel_representations, dim=1)
        
        else:
            # Standard patch aggregation
            v_flat = v.reshape(B, N, M * L, 1)
            t_flat = t.reshape(B, N, M * L, 1)
            mask_flat = mask.reshape(B, N, M * L, 1)
            
            h = self.value_encoder(v_flat) * self.weight_net(t_flat)
            a = self.value_encoder(v_flat) + self.weight_net(t_flat)
            a = a * mask_flat + (1 - mask_flat) * (-1e8)
            a = F.softmax(a, dim=2)
            z = torch.sum(a * h, dim=2)
        
        return z


# ============================================================================
# Main Model
# ============================================================================

class MixerBlock(nn.Module):
    def __init__(self, d_model, n_channels, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.n_channels = n_channels
        
        # Channel mixing with pre-normalization
        self.channel_norm = nn.LayerNorm(d_model)
        self.channel_mlp = nn.Sequential(
            nn.Linear(n_channels, n_channels * 2),
            nn.GELU(),
            nn.Dropout(0.1),  # Reverted to 0.1 - sweet spot
            nn.Linear(n_channels * 2, n_channels),
            nn.Dropout(0.1),
        )
        
        # Token mixing with pre-normalization
        self.token_norm = nn.LayerNorm(d_model)
        self.token_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),  # Reverted to 0.1
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1),
        )
    
    def forward(self, x):
        # x: (B, N, d_model)
        
        # Channel mixing with pre-norm and residual
        residual = x
        x_norm = self.channel_norm(x)  # Pre-normalize
        x_transposed = x_norm.transpose(1, 2)  # (B, d_model, N)
        x_mixed = self.channel_mlp(x_transposed)
        x = residual + x_mixed.transpose(1, 2)  # Residual connection
        
        # Token mixing with pre-norm and residual
        residual = x
        x_norm = self.token_norm(x)  # Pre-normalize
        x = residual + self.token_mlp(x_norm)  # Residual connection
        
        return x


# ============================================================================
# Phase 2 #5: Enhanced Decoder with Attention
# ============================================================================

class AttentionDecoder(nn.Module):
    """Enhanced decoder with multi-head attention mechanism for better temporal fusion"""
    def __init__(self, d_in, d_out, n_heads=4):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            d_in, n_heads, dropout=0.1, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_in)
        
        # Feed-forward network with expansion
        self.ffn = nn.Sequential(
            nn.Linear(d_in, d_in * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_in * 2, d_out),
            nn.Dropout(0.1),
        )
        self.norm2 = nn.LayerNorm(d_in)
        
        # Final output projection
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
# Phase 1: Learnable Positional Encoding
# ============================================================================

        
        # Create random mask: keep_prob = 1 - drop_prob
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (B, 1, 1, ...)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_mask = torch.floor(random_tensor)  # 0 or 1
        
        # Scale residual to maintain expected value
        output = x + residual * binary_mask / keep_prob
        return output


# ============================================================================
# Phase 1: Learnable Positional Encoding
# ============================================================================

class LearnablePositionalEncoding(nn.Module):
    """Adds positional information to help model understand temporal order"""
    def __init__(self, d_model, max_len=256):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x):
        # x: (B, N, d_model)
        B, N, d_model = x.shape
        if N > self.pos_embedding.shape[1]:
            # Interpolate if sequence longer than max_len
            pos_emb = F.interpolate(
                self.pos_embedding.transpose(1, 2),
                size=N,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        else:
            pos_emb = self.pos_embedding[:, :N, :]
        
        return x + pos_emb


class IMTS_Mixer_FastRNN(nn.Module):
    """IMTS_Mixer with FastRNN-based segmentation"""
    
    def __init__(self, args):
        super().__init__()
        self.n_channels = args.ndim
        self.d_model = args.hid_dim
        self.d_time = args.te_dim
        self.n_layers = args.nlayer
        self.d_out = args.d_out if hasattr(args, "d_out") else self.d_model
        self.n_heads = getattr(args, "n_heads", 4)
        
        # Enable/disable segmentation
        self.use_segmentation = getattr(args, "use_segmentation", True)
        self.use_multiscale = getattr(args, "use_multiscale", True)
        
        # Aggregation modules with FastRNN
        self.patch_aggregation = FastRNNPatchAggregation(
            self.d_model, self.d_time, 
            use_segmentation=self.use_segmentation,
            use_multiscale=self.use_multiscale
        )
        self.channel_aggregation = FastRNNChannelAggregation(
            self.d_model, self.d_time,
            use_segmentation=self.use_segmentation,
            use_multiscale=self.use_multiscale
        )
        
        self.channel_bias = nn.Parameter(torch.randn(1, self.n_channels, self.d_model))
        
        # ADD: Positional encoding for temporal awareness
        self.pos_encoding = LearnablePositionalEncoding(self.d_model, max_len=256)
        
        # Pre-normalization layers for each mixer block
        self.pre_norms = nn.ModuleList([
            nn.LayerNorm(self.d_model) for _ in range(self.n_layers)
        ])
        
        # Mixer blocks
        self.mixer_blocks = nn.ModuleList([
            MixerBlock(self.d_model, self.n_channels, self.n_heads)
            for _ in range(self.n_layers)
        ])
        
        # Final normalization after all mixer blocks
        self.final_norm = nn.LayerNorm(self.d_out if self.d_model != self.d_out else self.d_model)
        
        # Output projection with dropout
        if self.d_model != self.d_out:
            self.output_projection = nn.Sequential(
                nn.Linear(self.d_model, self.d_out),
                nn.Dropout(0.1),
            )
        else:
            self.output_projection = nn.Identity()
        
        # Phase 2 #5: Replace simple decoder with AttentionDecoder
        self.decoder = AttentionDecoder(self.d_out, self.d_out, n_heads=4)
        
        # Time encoder with dropout
        self.time_encoder_pred = nn.Sequential(
            nn.Linear(1, self.d_time),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_time, self.d_out),
        )
        
        # Simple weight initialization (Phase 1 style)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights properly for stable training"""
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
    
        # Initialize temperature parameter for attention (if exists in aggregation)
        if hasattr(self.channel_aggregation, 'temperature'):
            nn.init.constant_(self.channel_aggregation.temperature, 0.1)
    
    def forecasting(self, tp_to_predict, observed_data, observed_tp, observed_mask):
        """
        Forecasting with FastRNN segmentation support
        """
        # Detect data format
        if len(observed_data.shape) == 4:
            # Patched data: (B, M, L, N)
            B, M, L, N = observed_data.shape
            x = observed_data.permute(0, 3, 1, 2).unsqueeze(-1)
            t = observed_tp.permute(0, 3, 1, 2).unsqueeze(-1)
            mask = observed_mask.permute(0, 3, 1, 2).unsqueeze(-1)
            z = self.patch_aggregation(x, t, mask)
            is_patched = True
        else:
            # Non-patched data: (B, L, N)
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
        
        # Apply positional encoding for temporal awareness
        z = self.pos_encoding(z)
        
        # Apply mixer blocks with pre-normalization (Phase 1 style)
        for pre_norm, mixer_block in zip(self.pre_norms, self.mixer_blocks):
            z_norm = pre_norm(z)  # Pre-normalize
            z = z + mixer_block(z_norm)  # Residual connection
        
        # Project to output dimension
        z = self.output_projection(z)
        z = self.final_norm(z)  # Final normalization
        
        # Prepare for forecasting
        z = z.unsqueeze(-2)
        L_pred = tp_to_predict.shape[-1]
        z = z.repeat(1, 1, L_pred, 1)
        
        # Time encoding
        tp_to_predict = tp_to_predict.view(B, 1, L_pred, 1).repeat(1, N, 1, 1)
        te_pred = self.time_encoder_pred(tp_to_predict)
        
        # Combine and decode
        z = z + te_pred
        
        # Phase 2 #5: Reshape for AttentionDecoder (B, N, L_pred, d_out) -> (B*N, L_pred, d_out)
        B, N, L_pred, d_out = z.shape
        z_reshaped = z.view(B * N, L_pred, d_out)
        
        # Apply AttentionDecoder
        outputs = self.decoder(z_reshaped)  # (B*N, L_pred, 1)
        
        # Reshape back: (B*N, L_pred, 1) -> (B, N, L_pred)
        outputs = outputs.view(B, N, L_pred).unsqueeze(0).permute(0, 1, 3, 2)
        
        return outputs
