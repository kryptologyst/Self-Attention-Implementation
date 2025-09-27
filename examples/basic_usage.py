"""
Basic Usage Examples for Self-Attention Implementation
====================================================

This file demonstrates basic usage patterns for the self-attention implementation.
"""

import torch
import sys
import os

# Add parent directory to path to import our implementation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attention_implementation import (
    SelfAttention, MultiHeadAttention, PositionalEncoding,
    AttentionConfig, AttentionVisualizer, create_causal_mask
)


def example_1_basic_single_head():
    """Example 1: Basic single-head self-attention"""
    print("Example 1: Basic Single-Head Self-Attention")
    print("=" * 50)
    
    # Create input tensor (batch_size=1, seq_len=4, embed_dim=8)
    x = torch.randn(1, 4, 8)
    print(f"Input shape: {x.shape}")
    
    # Create self-attention layer
    attention = SelfAttention(embed_dim=8)
    
    # Apply attention
    output, attn_weights = attention(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Attention weights:\n{attn_weights.squeeze(0).detach().numpy()}")
    
    return output, attn_weights


def example_2_multi_head_attention():
    """Example 2: Multi-head self-attention"""
    print("\nExample 2: Multi-Head Self-Attention")
    print("=" * 50)
    
    # Configuration
    config = AttentionConfig(embed_dim=64, num_heads=8)
    
    # Create input tensor
    x = torch.randn(1, 8, 64)
    print(f"Input shape: {x.shape}")
    
    # Create multi-head attention
    multi_attention = MultiHeadAttention(config)
    
    # Apply attention
    output, attn_weights = multi_attention(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Number of heads: {config.num_heads}")
    
    return output, attn_weights


def example_3_with_causal_mask():
    """Example 3: Self-attention with causal mask"""
    print("\nExample 3: Self-Attention with Causal Mask")
    print("=" * 50)
    
    seq_len = 6
    embed_dim = 16
    
    # Create input
    x = torch.randn(1, seq_len, embed_dim)
    
    # Create causal mask
    mask = create_causal_mask(seq_len)
    print(f"Causal mask shape: {mask.shape}")
    print(f"Causal mask:\n{mask.squeeze(0).numpy()}")
    
    # Create attention layer
    attention = SelfAttention(embed_dim)
    
    # Apply attention with mask
    output, attn_weights = attention(x, mask=mask)
    
    print(f"Attention weights with causal mask:\n{attn_weights.squeeze(0).detach().numpy()}")
    
    return output, attn_weights


def example_4_positional_encoding():
    """Example 4: Positional encoding"""
    print("\nExample 4: Positional Encoding")
    print("=" * 50)
    
    embed_dim = 32
    seq_len = 10
    
    # Create positional encoding
    pos_enc = PositionalEncoding(embed_dim)
    
    # Create dummy input (seq_len, batch_size, embed_dim)
    x = torch.randn(seq_len, 1, embed_dim)
    print(f"Input shape: {x.shape}")
    
    # Add positional encoding
    x_with_pos = pos_enc(x)
    print(f"Output shape: {x_with_pos.shape}")
    
    # Show positional encoding matrix
    pe_matrix = pos_enc.pe[:seq_len, 0, :].numpy()
    print(f"Positional encoding matrix shape: {pe_matrix.shape}")
    print(f"First few values:\n{pe_matrix[:3, :5]}")
    
    return x_with_pos


def example_5_attention_visualization():
    """Example 5: Attention visualization"""
    print("\nExample 5: Attention Visualization")
    print("=" * 50)
    
    # Create sample text tokens
    tokens = ["The", "quick", "brown", "fox", "jumps"]
    seq_len = len(tokens)
    embed_dim = 32
    
    # Create input
    x = torch.randn(1, seq_len, embed_dim)
    
    # Create attention layer
    attention = SelfAttention(embed_dim)
    
    # Apply attention
    output, attn_weights = attention(x)
    
    print(f"Tokens: {tokens}")
    print(f"Attention weights:\n{attn_weights.squeeze(0).detach().numpy()}")
    
    # Note: Visualization would require matplotlib/seaborn
    # visualizer = AttentionVisualizer()
    # visualizer.plot_attention_weights(attn_weights, tokens)
    
    return output, attn_weights


def example_6_gradient_flow():
    """Example 6: Gradient flow demonstration"""
    print("\nExample 6: Gradient Flow Demonstration")
    print("=" * 50)
    
    embed_dim = 16
    seq_len = 4
    
    # Create input with requires_grad=True
    x = torch.randn(1, seq_len, embed_dim, requires_grad=True)
    
    # Create attention layer
    attention = SelfAttention(embed_dim)
    
    # Apply attention
    output, attn_weights = attention(x)
    
    # Compute loss and backpropagate
    loss = output.sum()
    loss.backward()
    
    print(f"Input gradient shape: {x.grad.shape}")
    print(f"Input gradient norm: {x.grad.norm().item():.4f}")
    print(f"Loss: {loss.item():.4f}")
    
    return output, attn_weights


def main():
    """Run all examples"""
    print("🚀 Self-Attention Implementation Examples")
    print("=" * 60)
    
    # Run examples
    example_1_basic_single_head()
    example_2_multi_head_attention()
    example_3_with_causal_mask()
    example_4_positional_encoding()
    example_5_attention_visualization()
    example_6_gradient_flow()
    
    print("\n✅ All examples completed successfully!")


if __name__ == "__main__":
    main()
