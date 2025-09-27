"""
Visualization Demo for Self-Attention Implementation
===================================================

This file demonstrates various visualization techniques for attention patterns.
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attention_implementation import (
    SelfAttention, MultiHeadAttention, PositionalEncoding,
    AttentionConfig, AttentionVisualizer
)


def create_sample_embeddings(text: str, embed_dim: int = 64) -> tuple:
    """Create sample embeddings from text"""
    tokens = text.split()
    seq_len = len(tokens)
    
    # Create random embeddings (in practice, these would come from a trained model)
    embeddings = torch.randn(1, seq_len, embed_dim)
    
    return embeddings, tokens


def demo_single_head_visualization():
    """Demonstrate single-head attention visualization"""
    print("Single-Head Attention Visualization")
    print("=" * 40)
    
    # Sample text
    text = "The quick brown fox jumps over the lazy dog"
    embed_dim = 64
    
    # Create embeddings and tokens
    embeddings, tokens = create_sample_embeddings(text, embed_dim)
    
    # Create attention layer
    attention = SelfAttention(embed_dim)
    
    # Apply attention
    output, attn_weights = attention(embeddings)
    
    # Visualize attention weights
    visualizer = AttentionVisualizer()
    visualizer.plot_attention_weights(
        attn_weights, 
        tokens, 
        "Single-Head Attention Weights"
    )
    
    return attn_weights, tokens


def demo_multi_head_visualization():
    """Demonstrate multi-head attention visualization"""
    print("\nMulti-Head Attention Visualization")
    print("=" * 40)
    
    # Sample text
    text = "Attention is all you need for transformers"
    embed_dim = 64
    num_heads = 8
    
    # Create embeddings and tokens
    embeddings, tokens = create_sample_embeddings(text, embed_dim)
    
    # Create multi-head attention
    config = AttentionConfig(embed_dim=embed_dim, num_heads=num_heads)
    multi_attention = MultiHeadAttention(config)
    
    # Apply attention
    output, attn_weights = multi_attention(embeddings)
    
    # Visualize multi-head attention
    visualizer = AttentionVisualizer()
    visualizer.plot_multihead_attention(attn_weights, tokens)
    
    return attn_weights, tokens


def demo_positional_encoding_visualization():
    """Demonstrate positional encoding visualization"""
    print("\nPositional Encoding Visualization")
    print("=" * 40)
    
    embed_dim = 64
    seq_len = 20
    
    # Create positional encoding
    pos_enc = PositionalEncoding(embed_dim)
    
    # Visualize positional encoding matrix
    pe_matrix = pos_enc.pe[:seq_len, 0, :].numpy()
    
    plt.figure(figsize=(12, 8))
    plt.imshow(pe_matrix.T, aspect='auto', cmap='RdYlBu')
    plt.colorbar()
    plt.title('Positional Encoding Matrix')
    plt.xlabel('Position')
    plt.ylabel('Embedding Dimension')
    plt.show()
    
    return pe_matrix


def demo_attention_statistics():
    """Demonstrate attention statistics analysis"""
    print("\nAttention Statistics Analysis")
    print("=" * 40)
    
    # Sample text
    text = "Self attention allows each token to attend to all other tokens"
    embed_dim = 64
    
    # Create embeddings and tokens
    embeddings, tokens = create_sample_embeddings(text, embed_dim)
    
    # Create attention layer
    attention = SelfAttention(embed_dim)
    
    # Apply attention
    output, attn_weights = attention(embeddings)
    
    # Convert to numpy for analysis
    attn_np = attn_weights.squeeze(0).detach().cpu().numpy()
    
    # Calculate statistics
    print(f"Attention matrix shape: {attn_np.shape}")
    print(f"Max attention weight: {np.max(attn_np):.4f}")
    print(f"Min attention weight: {np.min(attn_np):.4f}")
    print(f"Mean attention weight: {np.mean(attn_np):.4f}")
    print(f"Std attention weight: {np.std(attn_np):.4f}")
    
    # Attention entropy (diversity measure)
    entropy = -np.sum(attn_np * np.log(attn_np + 1e-8), axis=1)
    print(f"\nAttention entropy per token:")
    for i, token in enumerate(tokens):
        print(f"  {token}: {entropy[i]:.4f}")
    
    # Plot attention distribution
    plt.figure(figsize=(10, 6))
    plt.hist(attn_np.flatten(), bins=20, alpha=0.7, edgecolor='black')
    plt.title('Attention Weight Distribution')
    plt.xlabel('Attention Weight')
    plt.ylabel('Frequency')
    plt.show()
    
    # Plot attention entropy
    plt.figure(figsize=(10, 6))
    plt.bar(tokens, entropy)
    plt.title('Attention Entropy per Token')
    plt.xlabel('Token')
    plt.ylabel('Entropy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return attn_np, entropy


def demo_attention_patterns():
    """Demonstrate different attention patterns"""
    print("\nAttention Patterns Analysis")
    print("=" * 40)
    
    seq_len = 8
    embed_dim = 32
    
    # Create different types of inputs to see different attention patterns
    patterns = {
        "Random": torch.randn(1, seq_len, embed_dim),
        "Sequential": torch.arange(seq_len).float().unsqueeze(0).unsqueeze(-1).expand(1, seq_len, embed_dim),
        "Repeated": torch.zeros(1, seq_len, embed_dim).fill_(0.5),
    }
    
    attention = SelfAttention(embed_dim)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (pattern_name, x) in enumerate(patterns.items()):
        output, attn_weights = attention(x)
        attn_np = attn_weights.squeeze(0).detach().cpu().numpy()
        
        sns.heatmap(attn_np, annot=True, fmt='.2f', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'{pattern_name} Pattern')
        axes[i].set_xlabel('Key Position')
        axes[i].set_ylabel('Query Position')
    
    plt.tight_layout()
    plt.show()


def main():
    """Run all visualization demos"""
    print("🎨 Self-Attention Visualization Demo")
    print("=" * 50)
    
    # Run visualization demos
    demo_single_head_visualization()
    demo_multi_head_visualization()
    demo_positional_encoding_visualization()
    demo_attention_statistics()
    demo_attention_patterns()
    
    print("\n✅ All visualization demos completed!")


if __name__ == "__main__":
    main()
