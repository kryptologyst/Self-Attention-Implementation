"""
Project 129: Advanced Self-Attention Implementation
==================================================

A comprehensive implementation of self-attention mechanisms including:
- Single-head self-attention
- Multi-head self-attention
- Positional encoding
- Attention visualization tools
- Performance benchmarking

This project demonstrates the core building blocks of Transformer architectures
using modern PyTorch best practices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, Union
import time
from dataclasses import dataclass


@dataclass
class AttentionConfig:
    """Configuration for attention mechanisms"""
    embed_dim: int = 512
    num_heads: int = 8
    dropout: float = 0.1
    max_seq_len: int = 1024
    bias: bool = True


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence position information"""
    
    def __init__(self, embed_dim: int, max_seq_len: int = 5000):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, embed_dim)
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:x.size(0), :]


class SelfAttention(nn.Module):
    """Single-head self-attention mechanism"""
    
    def __init__(self, embed_dim: int, dropout: float = 0.1, bias: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = math.sqrt(embed_dim)
        
        # Linear projections for Q, K, V
        self.query = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.key = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.value = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional attention mask of shape (batch_size, seq_len, seq_len)
        Returns:
            output: Contextualized embeddings
            attn_weights: Attention weights for visualization
        """
        batch_size, seq_len, embed_dim = x.size()
        
        # Compute Q, K, V
        Q = self.query(x)  # (B, L, D)
        K = self.key(x)    # (B, L, D)
        V = self.value(x)  # (B, L, D)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        output = self.out_proj(context)
        
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        assert config.embed_dim % config.num_heads == 0
        
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Linear projections
        self.q_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional attention mask
        Returns:
            output: Multi-head attention output
            attn_weights: Attention weights for visualization
        """
        batch_size, seq_len, embed_dim = x.size()
        
        # Linear projections
        Q = self.q_proj(x)  # (B, L, D)
        K = self.k_proj(x)  # (B, L, D)
        V = self.v_proj(x)  # (B, L, D)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D/H)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D/H)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D/H)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)  # (B, H, L, D/H)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # Final linear projection
        output = self.out_proj(context)
        
        return output, attn_weights


class AttentionVisualizer:
    """Tools for visualizing attention patterns"""
    
    @staticmethod
    def plot_attention_weights(attn_weights: torch.Tensor, tokens: Optional[list] = None, 
                              title: str = "Attention Weights", figsize: tuple = (10, 8)):
        """Plot attention weights as a heatmap"""
        attn_np = attn_weights.squeeze(0).detach().cpu().numpy()
        
        plt.figure(figsize=figsize)
        sns.heatmap(attn_np, annot=True, fmt='.3f', cmap='Blues', 
                   xticklabels=tokens, yticklabels=tokens)
        plt.title(title)
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_multihead_attention(attn_weights: torch.Tensor, tokens: Optional[list] = None,
                               figsize: tuple = (15, 10)):
        """Plot multi-head attention weights"""
        batch_size, num_heads, seq_len, _ = attn_weights.shape
        attn_np = attn_weights.squeeze(0).detach().cpu().numpy()
        
        fig, axes = plt.subplots(2, (num_heads + 1) // 2, figsize=figsize)
        axes = axes.flatten() if num_heads > 1 else [axes]
        
        for head in range(num_heads):
            sns.heatmap(attn_np[head], annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=tokens, yticklabels=tokens, ax=axes[head])
            axes[head].set_title(f'Head {head + 1}')
            axes[head].set_xlabel('Key Position')
            axes[head].set_ylabel('Query Position')
        
        plt.suptitle('Multi-Head Attention Weights')
        plt.tight_layout()
        plt.show()


class AttentionBenchmark:
    """Performance benchmarking for attention mechanisms"""
    
    @staticmethod
    def benchmark_attention(model: nn.Module, input_shape: tuple, 
                          num_runs: int = 100, warmup_runs: int = 10) -> dict:
        """Benchmark attention model performance"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        # Create test input
        x = torch.randn(input_shape).to(device)
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(x)
        
        # Benchmark runs
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(x)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        throughput = input_shape[0] * input_shape[1] / avg_time
        
        return {
            'avg_time_ms': avg_time * 1000,
            'throughput_tokens_per_sec': throughput,
            'device': str(device),
            'input_shape': input_shape
        }


def create_causal_mask(seq_len: int) -> torch.Tensor:
    """Create causal mask for autoregressive attention"""
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0)


def demo_single_head_attention():
    """Demonstrate single-head self-attention"""
    print("🔍 Single-Head Self-Attention Demo")
    print("=" * 50)
    
    # Configuration
    batch_size, seq_len, embed_dim = 1, 8, 64
    tokens = [f"token_{i}" for i in range(seq_len)]
    
    # Create input
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Create and apply self-attention
    self_attn = SelfAttention(embed_dim=embed_dim)
    output, attn_weights = self_attn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    
    # Visualize attention
    visualizer = AttentionVisualizer()
    visualizer.plot_attention_weights(attn_weights, tokens, "Single-Head Attention")
    
    return output, attn_weights


def demo_multi_head_attention():
    """Demonstrate multi-head self-attention"""
    print("\n🔍 Multi-Head Self-Attention Demo")
    print("=" * 50)
    
    # Configuration
    config = AttentionConfig(embed_dim=64, num_heads=8)
    batch_size, seq_len = 1, 8
    tokens = [f"token_{i}" for i in range(seq_len)]
    
    # Create input
    x = torch.randn(batch_size, seq_len, config.embed_dim)
    
    # Create and apply multi-head attention
    multi_attn = MultiHeadAttention(config)
    output, attn_weights = multi_attn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Number of heads: {config.num_heads}")
    
    # Visualize multi-head attention
    visualizer = AttentionVisualizer()
    visualizer.plot_multihead_attention(attn_weights, tokens)
    
    return output, attn_weights


def demo_positional_encoding():
    """Demonstrate positional encoding"""
    print("\n📍 Positional Encoding Demo")
    print("=" * 50)
    
    embed_dim, seq_len = 64, 16
    pos_enc = PositionalEncoding(embed_dim)
    
    # Create dummy input
    x = torch.randn(seq_len, 1, embed_dim)  # (seq_len, batch_size, embed_dim)
    
    # Add positional encoding
    x_with_pos = pos_enc(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {x_with_pos.shape}")
    
    # Visualize positional encoding
    pe_matrix = pos_enc.pe[:seq_len, 0, :].numpy()
    
    plt.figure(figsize=(12, 8))
    plt.imshow(pe_matrix.T, aspect='auto', cmap='RdYlBu')
    plt.colorbar()
    plt.title('Positional Encoding Matrix')
    plt.xlabel('Position')
    plt.ylabel('Embedding Dimension')
    plt.show()
    
    return x_with_pos


def benchmark_comparison():
    """Compare performance of different attention mechanisms"""
    print("\n⚡ Performance Benchmark")
    print("=" * 50)
    
    benchmark = AttentionBenchmark()
    
    # Test configurations
    configs = [
        (SelfAttention(64), (1, 64, 64), "Single-Head"),
        (MultiHeadAttention(AttentionConfig(64, 8)), (1, 64, 64), "Multi-Head (8 heads)"),
        (MultiHeadAttention(AttentionConfig(64, 16)), (1, 64, 64), "Multi-Head (16 heads)"),
    ]
    
    results = []
    for model, input_shape, name in configs:
        result = benchmark.benchmark_attention(model, input_shape)
        result['model_name'] = name
        results.append(result)
        
        print(f"{name}:")
        print(f"  Average time: {result['avg_time_ms']:.2f} ms")
        print(f"  Throughput: {result['throughput_tokens_per_sec']:.0f} tokens/sec")
        print(f"  Device: {result['device']}")
        print()
    
    return results


def main():
    """Main demonstration function"""
    print("🚀 Advanced Self-Attention Implementation Demo")
    print("=" * 60)
    
    # Run demonstrations
    demo_single_head_attention()
    demo_multi_head_attention()
    demo_positional_encoding()
    benchmark_comparison()
    
    print("\n✅ All demonstrations completed!")
    print("\n🧠 What This Project Demonstrates:")
    print("• Implements the core of self-attention used in every transformer layer")
    print("• Shows how every token attends to every other token in a sequence")
    print("• Produces contextualized token embeddings with learned importance weights")
    print("• Multi-head attention for parallel attention computation")
    print("• Positional encoding for sequence position information")
    print("• Performance benchmarking and visualization tools")


if __name__ == "__main__":
    main()