#!/usr/bin/env python3
"""
Test script for enhanced residual connections in APNTSMixer
"""
import sys
sys.path.append('t-PatchGNN/tPatchGNN')

import torch
import argparse
import time
from model.APNTSMixer import APNTSMixer

def test_enhanced_residuals():
    """Test the enhanced residual connections"""
    
    # Create mock arguments for testing
    args = argparse.Namespace()
    args.npatch = 32  # Increased for better performance
    args.hid_dim = 128  # Increased hidden dimension
    args.te_dim = 10
    args.ndim = 41  # physionet dimensions
    args.t_obs = 24
    args.nlayer = 4  # More layers to benefit from residuals
    args.dropout = 0.15
    args.expansion_factor = 4  # Larger expansion factor
    args.use_attention = True
    args.use_adaptive_loss = True
    args.adaptive_loss_type = "multi_component"
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Testing on device: {args.device}")
    print("=== Testing Enhanced Residual Connections ===")
    
    # Create model
    model = APNTSMixer(args).to(args.device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass with different batch sizes
    batch_sizes = [1, 4, 8]
    seq_len = 24
    pred_len = 12
    
    for batch_size in batch_sizes:
        print(f"\n--- Testing batch size: {batch_size} ---")
        
        # Create dummy data
        x = torch.randn(batch_size, seq_len, args.ndim).to(args.device)
        t = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1, args.ndim).float().to(args.device)
        mask = torch.ones_like(x).to(args.device)
        t_pred = torch.arange(seq_len, seq_len + pred_len).unsqueeze(0).repeat(batch_size, 1).float().to(args.device)
        y_true = torch.randn(batch_size, pred_len, args.ndim).to(args.device)
        
        # Time the forward pass
        start_time = time.time()
        
        with torch.no_grad():
            pred_y = model.forecasting(t_pred, x, t, mask)
        
        forward_time = time.time() - start_time
        
        print(f"Input shape: {x.shape}")
        print(f"Prediction shape: {pred_y.shape}")
        print(f"Forward pass time: {forward_time:.4f}s")
        
        # Test loss computation
        if hasattr(model, 'use_adaptive_loss') and model.use_adaptive_loss:
            pred_for_loss = pred_y.squeeze(0) if len(pred_y.shape) == 4 else pred_y
            loss_result = model.compute_adaptive_loss(pred_for_loss, y_true)
            
            if isinstance(loss_result, dict):
                print(f"Adaptive loss: {loss_result['total_loss'].item():.6f}")
                
                # Check residual weights
                if hasattr(model, 'mixer_layers'):
                    for i, layer in enumerate(model.mixer_layers):
                        if hasattr(layer, 'patch_residual_weight'):
                            print(f"Layer {i} patch residual weight: {layer.patch_residual_weight.item():.4f}")
                        if hasattr(layer, 'channel_residual_weight'):
                            print(f"Layer {i} channel residual weight: {layer.channel_residual_weight.item():.4f}")
            else:
                print(f"Loss: {loss_result.item():.6f}")
        else:
            # Standard MSE
            loss = torch.nn.functional.mse_loss(pred_y.squeeze(0), y_true)
            print(f"Standard MSE loss: {loss.item():.6f}")
    
    print("\n✓ Enhanced residual connections test passed!")
    
    # Test gradient flow
    print("\n=== Testing Gradient Flow ===")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Single training step
    x = torch.randn(2, seq_len, args.ndim).to(args.device)
    t = torch.arange(seq_len).unsqueeze(0).repeat(2, 1, args.ndim).float().to(args.device)
    mask = torch.ones_like(x).to(args.device)
    t_pred = torch.arange(seq_len, seq_len + pred_len).unsqueeze(0).repeat(2, 1).float().to(args.device)
    y_true = torch.randn(2, pred_len, args.ndim).to(args.device)
    
    optimizer.zero_grad()
    pred_y = model.forecasting(t_pred, x, t, mask)
    
    if hasattr(model, 'use_adaptive_loss') and model.use_adaptive_loss:
        pred_for_loss = pred_y.squeeze(0) if len(pred_y.shape) == 4 else pred_y
        loss_result = model.compute_adaptive_loss(pred_for_loss, y_true)
        loss = loss_result['total_loss'] if isinstance(loss_result, dict) else loss_result
    else:
        loss = torch.nn.functional.mse_loss(pred_y.squeeze(0), y_true)
    
    loss.backward()
    
    # Check gradient norms
    total_grad_norm = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_grad_norm = param.grad.data.norm(2)
            total_grad_norm += param_grad_norm.item() ** 2
            if 'residual_weight' in name:
                print(f"Gradient norm for {name}: {param_grad_norm:.6f}")
    
    total_grad_norm = total_grad_norm ** (1. / 2)
    print(f"Total gradient norm: {total_grad_norm:.6f}")
    
    optimizer.step()
    print("✓ Gradient flow test passed!")
    
if __name__ == "__main__":
    test_enhanced_residuals()
