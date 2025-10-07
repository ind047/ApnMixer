"""
IMTS_Mixer with 1D U-Net Segmentation
Extends the original IMTS_Mixer with intelligent U-Net-based segmentation for better temporal pa        d4 = self.up4(bottleneck)
        # Adjust size to match e4 if needed
        if d4.size(2) != e4.size(2):
            d4 = F.interpolate(d4, size=e4.size(2), mode='linear', align_corners=False)
        e4_att = self.att4(d4, e4)
        d4 = torch.cat([d4, e4_att], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        # Adjust size to match e3 if needed
        if d3.size(2) != e3.size(2):
            d3 = F.interpolate(d3, size=e3.size(2), mode='linear', align_corners=False)
        e3_att = self.att3(d3, e3)
        d3 = torch.cat([d3, e3_att], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        # Adjust size to match e2 if needed
        if d2.size(2) != e2.size(2):
            d2 = F.interpolate(d2, size=e2.size(2), mode='linear', align_corners=False)
        e2_att = self.att2(d2, e2)
        d2 = torch.cat([d2, e2_att], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        # Adjust size to match e1 if needed
        if d1.size(2) != e1.size(2):
            d1 = F.interpolate(d1, size=e1.size(2), mode='linear', align_corners=False)
        e1_att = self.att1(d1, e1)
        d1 = torch.cat([d1, e1_att], dim=1)
        d1 = self.dec1(d1).
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
# U-Net Components
# ============================================================================

class Conv1DBlock(nn.Module):
    """Basic convolutional block for U-Net"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self.norm2(self.conv2(out)))
        return out


class AttentionGate1D(nn.Module):
    """Attention gate for U-Net skip connections"""
    def __init__(self, gate_channels, in_channels, inter_channels=None):
        super().__init__()
        
        if inter_channels is None:
            inter_channels = in_channels // 2
        
        self.W_g = nn.Conv1d(gate_channels, inter_channels, 1)
        self.W_x = nn.Conv1d(in_channels, inter_channels, 1)
        self.psi = nn.Conv1d(inter_channels, 1, 1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Resize g1 to match x1 if needed
        if g1.shape[-1] != x1.shape[-1]:
            g1 = F.interpolate(g1, size=x1.shape[-1], mode='linear', align_corners=False)
        
        psi = self.relu(g1 + x1)
        psi = torch.sigmoid(self.psi(psi))
        
        return x * psi


class UNet1DSegmenter(nn.Module):
    """1D U-Net for time series segmentation"""
    def __init__(self, input_dim=1, base_channels=16, n_segment_types=5):
        super().__init__()
        
        # Encoder (downsampling path)
        self.enc1 = Conv1DBlock(input_dim, base_channels)
        self.enc2 = Conv1DBlock(base_channels, base_channels*2)
        self.enc3 = Conv1DBlock(base_channels*2, base_channels*4)
        self.enc4 = Conv1DBlock(base_channels*4, base_channels*8)
        
        # Bottleneck
        self.bottleneck = Conv1DBlock(base_channels*8, base_channels*16)
        
        # Decoder (upsampling path)
        self.up4 = nn.ConvTranspose1d(base_channels*16, base_channels*8, 2, stride=2)
        self.att4 = AttentionGate1D(base_channels*8, base_channels*8)
        self.dec4 = Conv1DBlock(base_channels*16, base_channels*8)
        
        self.up3 = nn.ConvTranspose1d(base_channels*8, base_channels*4, 2, stride=2)
        self.att3 = AttentionGate1D(base_channels*4, base_channels*4)
        self.dec3 = Conv1DBlock(base_channels*8, base_channels*4)
        
        self.up2 = nn.ConvTranspose1d(base_channels*4, base_channels*2, 2, stride=2)
        self.att2 = AttentionGate1D(base_channels*2, base_channels*2)
        self.dec2 = Conv1DBlock(base_channels*4, base_channels*2)
        
        self.up1 = nn.ConvTranspose1d(base_channels*2, base_channels, 2, stride=2)
        self.att1 = AttentionGate1D(base_channels, base_channels)
        self.dec1 = Conv1DBlock(base_channels*2, base_channels)
        
        # Output heads
        self.segment_head = nn.Conv1d(base_channels, n_segment_types, 1)
        self.boundary_head = nn.Conv1d(base_channels, 1, 1)
        
        # Pooling
        self.pool = nn.MaxPool1d(2)
    
    def forward(self, x, return_features=False):
        # x: (B, L, D) -> (B, D, L) for conv1d
        x = x.transpose(1, 2)
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with attention
        d4 = self.up4(b)
        # Ensure d4 matches e4 size before attention and concatenation
        if d4.size(2) != e4.size(2):
            d4 = F.interpolate(d4, size=e4.size(2), mode='linear', align_corners=False)
        e4_att = self.att4(d4, e4)
        d4 = torch.cat([d4, e4_att], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        # Ensure d3 matches e3 size before attention and concatenation
        if d3.size(2) != e3.size(2):
            d3 = F.interpolate(d3, size=e3.size(2), mode='linear', align_corners=False)
        e3_att = self.att3(d3, e3)
        d3 = torch.cat([d3, e3_att], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        # Ensure d2 matches e2 size before attention and concatenation
        if d2.size(2) != e2.size(2):
            d2 = F.interpolate(d2, size=e2.size(2), mode='linear', align_corners=False)
        e2_att = self.att2(d2, e2)
        d2 = torch.cat([d2, e2_att], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        # Ensure d1 matches e1 size before attention and concatenation
        if d1.size(2) != e1.size(2):
            d1 = F.interpolate(d1, size=e1.size(2), mode='linear', align_corners=False)
        e1_att = self.att1(d1, e1)
        d1 = torch.cat([d1, e1_att], dim=1)
        d1 = self.dec1(d1)
        
        # Output heads
        segment_logits = self.segment_head(d1).transpose(1, 2)  # (B, L, n_types)
        boundary_probs = torch.sigmoid(self.boundary_head(d1)).transpose(1, 2).squeeze(-1)  # (B, L)
        
        if return_features:
            features = {
                'encoder': [e1, e2, e3, e4],
                'bottleneck': b,
                'decoder': [d1, d2, d3, d4]
            }
            return segment_logits, boundary_probs, features
        
        return segment_logits, boundary_probs
    
    def extract_segments(self, x, boundary_threshold=0.5):
        """Extract segments from U-Net predictions"""
        segment_logits, boundary_probs = self.forward(x)
        
        segments_list = []
        segment_types_list = []
        
        for b in range(x.shape[0]):
            # Find boundaries
            boundaries = torch.where(boundary_probs[b] > boundary_threshold)[0].cpu().numpy()
            
            # Convert to segment intervals
            if len(boundaries) == 0:
                segments = [(0, x.shape[1])]
            else:
                segments = []
                start = 0
                for boundary in boundaries:
                    if boundary > start:
                        segments.append((start, int(boundary)))
                        start = int(boundary)
                if start < x.shape[1]:
                    segments.append((start, x.shape[1]))
            
            # Get segment types
            types = []
            for start, end in segments:
                segment_logit = segment_logits[b, start:end].mean(dim=0)
                segment_type = torch.argmax(segment_logit).item()
                types.append(segment_type)
            
            segments_list.append(segments)
            segment_types_list.append(types)
        
        return segments_list, segment_types_list


# ============================================================================
# U-Net-based Aggregation Modules
# ============================================================================

class UNetChannelAggregation(nn.Module):
    """Channel aggregation with U-Net segmentation"""
    def __init__(self, d_model, d_time, use_segmentation=True):
        super().__init__()
        self.d_model = d_model
        self.use_segmentation = use_segmentation
        
        # U-Net segmenter
        if use_segmentation:
            self.segmenter = UNet1DSegmenter(input_dim=1, base_channels=16, n_segment_types=5)
        
        # Observation encoder
        self.observation_encoder = ObservationEncoder(d_model, d_time)
        self.weight_net = nn.Sequential(
            nn.Linear(1, d_time), nn.ReLU(inplace=True), nn.Linear(d_time, d_model)
        )
        self.value_encoder = nn.Linear(1, d_model)
        
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
        # t: (B, N, L, 1)
        # mask: (B, N, L, 1)
        
        B, N, L, _ = v.shape
        
        if self.use_segmentation:
            # Process each channel with segmentation
            channel_representations = []
            
            for n in range(N):
                channel_v = v[:, n]  # (B, L, 1)
                channel_t = t[:, n]  # (B, L, 1)
                channel_mask = mask[:, n]  # (B, L, 1)
                
                # Get segments from U-Net
                segments_list, segment_types_list = self.segmenter.extract_segments(channel_v)
                
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
                        seg_repr = torch.sum(a * h, dim=1)  # (1, d_model)
                        
                        # Apply segment-specific processing
                        seg_repr = self.segment_processors[str(seg_type)](seg_repr)
                        
                        segment_reprs.append(seg_repr)
                        segment_weights.append(end - start)
                    
                    if segment_reprs:
                        # Weighted combination
                        weights = torch.tensor(segment_weights, device=v.device, dtype=torch.float)
                        weights = F.softmax(weights, dim=0)
                        stacked = torch.stack(segment_reprs).squeeze(1)
                        batch_repr = torch.sum(weights.unsqueeze(-1) * stacked, dim=0)
                    else:
                        batch_repr = torch.zeros(self.d_model, device=v.device)
                    
                    batch_reprs.append(batch_repr)
                
                channel_repr = torch.stack(batch_reprs)  # (B, d_model)
                channel_representations.append(channel_repr)
            
            z = torch.stack(channel_representations, dim=1)  # (B, N, d_model)
        
        else:
            # Standard aggregation without segmentation
            h = self.value_encoder(v) * self.weight_net(t)
            a = self.value_encoder(v) + self.weight_net(t)
            a = a * mask + (1 - mask) * (-1e8)
            a = F.softmax(a, dim=2)
            z = torch.sum(a * h, dim=2)
        
        return z


class UNetPatchAggregation(nn.Module):
    """Patch aggregation with U-Net segmentation"""
    def __init__(self, d_model, d_time, use_segmentation=True):
        super().__init__()
        self.d_model = d_model
        self.use_segmentation = use_segmentation
        
        # U-Net segmenter
        if use_segmentation:
            self.segmenter = UNet1DSegmenter(input_dim=1, base_channels=16, n_segment_types=5)
        
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
            # Flatten patches and process with segmentation
            channel_representations = []
            
            for n in range(N):
                # Flatten patches for this channel
                channel_v = v[:, n].reshape(B, M * L, 1)  # (B, M*L, 1)
                channel_t = t[:, n].reshape(B, M * L, 1)
                channel_mask = mask[:, n].reshape(B, M * L, 1)
                
                # Get segments
                segments_list, segment_types_list = self.segmenter.extract_segments(channel_v)
                
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
            B, N, M, L, _ = v.shape
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
        self.channel_norm_old = RMSNorm(n_channels)
        self.channel_mlp = nn.Sequential(
            nn.Linear(n_channels, n_channels), nn.ReLU(inplace=True)
        )
        
        self.hidden_norm = RMSNorm(d_model)
        self.hidden_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        residual = x
        x = self.channel_norm_old(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.channel_mlp(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + residual
        
        residual = x
        x = self.hidden_norm(x)
        x = self.hidden_mlp(x)
        x = x + residual
        return x


class IMTS_Mixer_UNet(nn.Module):
    """IMTS_Mixer with U-Net-based segmentation"""
    
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
        
        # Aggregation modules with U-Net
        self.patch_aggregation = UNetPatchAggregation(
            self.d_model, self.d_time, use_segmentation=self.use_segmentation
        )
        self.channel_aggregation = UNetChannelAggregation(
            self.d_model, self.d_time, use_segmentation=self.use_segmentation
        )
        
        self.channel_bias = nn.Parameter(torch.randn(1, self.n_channels, self.d_model))
        
        # Mixer blocks
        self.mixer_blocks = nn.ModuleList([
            MixerBlock(self.d_model, self.n_channels, self.n_heads)
            for _ in range(self.n_layers)
        ])
        
        # Output projection
        if self.d_model != self.d_out:
            self.output_projection = nn.Linear(self.d_model, self.d_out)
        else:
            self.output_projection = nn.Identity()
        
        # Decoder
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
        Forecasting with U-Net segmentation support
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
        
        # Mixer blocks
        for mixer_block in self.mixer_blocks:
            z = mixer_block(z)
        
        # Project to output dimension
        z = self.output_projection(z)
        
        # Prepare for forecasting
        z = z.unsqueeze(-2)
        L_pred = tp_to_predict.shape[-1]
        z = z.repeat(1, 1, L_pred, 1)
        
        # Time encoding
        tp_to_predict = tp_to_predict.view(B, 1, L_pred, 1).repeat(1, N, 1, 1)
        te_pred = self.time_encoder_pred(tp_to_predict)
        
        # Combine and decode
        z = z + te_pred
        outputs = self.decoder(z).squeeze(dim=-1).permute(0, 2, 1).unsqueeze(dim=0)
        
        return outputs
