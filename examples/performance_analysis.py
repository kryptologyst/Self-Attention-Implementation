"""
Performance Analysis for Self-Attention Implementation
=====================================================

This file demonstrates performance benchmarking and analysis techniques.
"""

import torch
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attention_implementation import (
    SelfAttention, MultiHeadAttention, PositionalEncoding,
    AttentionConfig, AttentionBenchmark
)


def benchmark_scaling_analysis():
    """Analyze how performance scales with different parameters"""
    print("Performance Scaling Analysis")
    print("=" * 40)
    
    benchmark = AttentionBenchmark()
    
    # Test different sequence lengths
    seq_lengths = [16, 32, 64, 128, 256]
    embed_dim = 64
    
    results = []
    
    for seq_len in seq_lengths:
        print(f"Testing sequence length: {seq_len}")
        
        # Single-head attention
        single_head = SelfAttention(embed_dim)
        single_result = benchmark.benchmark_attention(
            single_head, (1, seq_len, embed_dim), num_runs=50, warmup_runs=5
        )
        single_result['model_type'] = 'Single-Head'
        single_result['seq_len'] = seq_len
        results.append(single_result)
        
        # Multi-head attention (8 heads)
        config = AttentionConfig(embed_dim=embed_dim, num_heads=8)
        multi_head = MultiHeadAttention(config)
        multi_result = benchmark.benchmark_attention(
            multi_head, (1, seq_len, embed_dim), num_runs=50, warmup_runs=5
        )
        multi_result['model_type'] = 'Multi-Head (8)'
        multi_result['seq_len'] = seq_len
        results.append(multi_result)
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(results)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot timing vs sequence length
    for model_type in df['model_type'].unique():
        data = df[df['model_type'] == model_type]
        ax1.plot(data['seq_len'], data['avg_time_ms'], marker='o', label=model_type)
    
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Average Time (ms)')
    ax1.set_title('Processing Time vs Sequence Length')
    ax1.legend()
    ax1.grid(True)
    
    # Plot throughput vs sequence length
    for model_type in df['model_type'].unique():
        data = df[df['model_type'] == model_type]
        ax2.plot(data['seq_len'], data['throughput_tokens_per_sec'], marker='o', label=model_type)
    
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Throughput (tokens/sec)')
    ax2.set_title('Throughput vs Sequence Length')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return df


def benchmark_head_analysis():
    """Analyze performance with different numbers of heads"""
    print("\nMulti-Head Performance Analysis")
    print("=" * 40)
    
    benchmark = AttentionBenchmark()
    
    # Test different numbers of heads
    num_heads_list = [1, 2, 4, 8, 16]
    embed_dim = 64
    seq_len = 64
    
    results = []
    
    for num_heads in num_heads_list:
        print(f"Testing {num_heads} heads")
        
        if embed_dim % num_heads != 0:
            print(f"Skipping {num_heads} heads (embed_dim not divisible)")
            continue
        
        config = AttentionConfig(embed_dim=embed_dim, num_heads=num_heads)
        multi_head = MultiHeadAttention(config)
        
        result = benchmark.benchmark_attention(
            multi_head, (1, seq_len, embed_dim), num_runs=50, warmup_runs=5
        )
        result['num_heads'] = num_heads
        results.append(result)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot timing vs number of heads
    ax1.plot(df['num_heads'], df['avg_time_ms'], marker='o')
    ax1.set_xlabel('Number of Heads')
    ax1.set_ylabel('Average Time (ms)')
    ax1.set_title('Processing Time vs Number of Heads')
    ax1.grid(True)
    
    # Plot throughput vs number of heads
    ax2.plot(df['num_heads'], df['throughput_tokens_per_sec'], marker='o')
    ax2.set_xlabel('Number of Heads')
    ax2.set_ylabel('Throughput (tokens/sec)')
    ax2.set_title('Throughput vs Number of Heads')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return df


def benchmark_embedding_dimension_analysis():
    """Analyze performance with different embedding dimensions"""
    print("\nEmbedding Dimension Performance Analysis")
    print("=" * 40)
    
    benchmark = AttentionBenchmark()
    
    # Test different embedding dimensions
    embed_dims = [32, 64, 128, 256, 512]
    seq_len = 64
    
    results = []
    
    for embed_dim in embed_dims:
        print(f"Testing embedding dimension: {embed_dim}")
        
        # Single-head attention
        single_head = SelfAttention(embed_dim)
        single_result = benchmark.benchmark_attention(
            single_head, (1, seq_len, embed_dim), num_runs=50, warmup_runs=5
        )
        single_result['model_type'] = 'Single-Head'
        single_result['embed_dim'] = embed_dim
        results.append(single_result)
        
        # Multi-head attention (8 heads)
        if embed_dim % 8 == 0:
            config = AttentionConfig(embed_dim=embed_dim, num_heads=8)
            multi_head = MultiHeadAttention(config)
            multi_result = benchmark.benchmark_attention(
                multi_head, (1, seq_len, embed_dim), num_runs=50, warmup_runs=5
            )
            multi_result['model_type'] = 'Multi-Head (8)'
            multi_result['embed_dim'] = embed_dim
            results.append(multi_result)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot timing vs embedding dimension
    for model_type in df['model_type'].unique():
        data = df[df['model_type'] == model_type]
        ax1.plot(data['embed_dim'], data['avg_time_ms'], marker='o', label=model_type)
    
    ax1.set_xlabel('Embedding Dimension')
    ax1.set_ylabel('Average Time (ms)')
    ax1.set_title('Processing Time vs Embedding Dimension')
    ax1.legend()
    ax1.grid(True)
    
    # Plot throughput vs embedding dimension
    for model_type in df['model_type'].unique():
        data = df[df['model_type'] == model_type]
        ax2.plot(data['embed_dim'], data['throughput_tokens_per_sec'], marker='o', label=model_type)
    
    ax2.set_xlabel('Embedding Dimension')
    ax2.set_ylabel('Throughput (tokens/sec)')
    ax2.set_title('Throughput vs Embedding Dimension')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return df


def memory_usage_analysis():
    """Analyze memory usage patterns"""
    print("\nMemory Usage Analysis")
    print("=" * 40)
    
    import psutil
    import gc
    
    def get_memory_usage():
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    # Test memory usage with different configurations
    configs = [
        (32, 8, "Small"),
        (64, 8, "Medium"),
        (128, 8, "Large"),
        (256, 8, "XLarge"),
    ]
    
    memory_results = []
    
    for embed_dim, num_heads, size_name in configs:
        print(f"Testing {size_name} configuration (embed_dim={embed_dim}, heads={num_heads})")
        
        # Clear memory
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        initial_memory = get_memory_usage()
        
        # Create model
        config = AttentionConfig(embed_dim=embed_dim, num_heads=num_heads)
        model = MultiHeadAttention(config)
        
        # Create input
        x = torch.randn(1, 64, embed_dim)
        
        # Forward pass
        output, attn_weights = model(x)
        
        after_forward_memory = get_memory_usage()
        
        memory_results.append({
            'size': size_name,
            'embed_dim': embed_dim,
            'num_heads': num_heads,
            'initial_memory_mb': initial_memory,
            'after_forward_memory_mb': after_forward_memory,
            'memory_increase_mb': after_forward_memory - initial_memory
        })
        
        # Clean up
        del model, x, output, attn_weights
    
    # Convert to DataFrame
    df = pd.DataFrame(memory_results)
    
    # Plot memory usage
    plt.figure(figsize=(10, 6))
    plt.bar(df['size'], df['memory_increase_mb'])
    plt.xlabel('Configuration Size')
    plt.ylabel('Memory Increase (MB)')
    plt.title('Memory Usage by Configuration')
    plt.show()
    
    print("\nMemory Usage Results:")
    print(df[['size', 'embed_dim', 'num_heads', 'memory_increase_mb']])
    
    return df


def comprehensive_performance_report():
    """Generate a comprehensive performance report"""
    print("\nComprehensive Performance Report")
    print("=" * 50)
    
    # Run all analyses
    scaling_df = benchmark_scaling_analysis()
    heads_df = benchmark_head_analysis()
    embed_df = benchmark_embedding_dimension_analysis()
    memory_df = memory_usage_analysis()
    
    # Generate summary statistics
    print("\n📊 Performance Summary:")
    print(f"Device: {scaling_df['device'].iloc[0]}")
    print(f"Tested sequence lengths: {sorted(scaling_df['seq_len'].unique())}")
    print(f"Tested embedding dimensions: {sorted(embed_df['embed_dim'].unique())}")
    print(f"Tested number of heads: {sorted(heads_df['num_heads'].unique())}")
    
    # Find optimal configurations
    best_throughput_single = scaling_df[scaling_df['model_type'] == 'Single-Head'].loc[
        scaling_df[scaling_df['model_type'] == 'Single-Head']['throughput_tokens_per_sec'].idxmax()
    ]
    
    best_throughput_multi = scaling_df[scaling_df['model_type'] == 'Multi-Head (8)'].loc[
        scaling_df[scaling_df['model_type'] == 'Multi-Head (8)']['throughput_tokens_per_sec'].idxmax()
    ]
    
    print(f"\n🏆 Best Performance:")
    print(f"Single-Head: {best_throughput_single['throughput_tokens_per_sec']:.0f} tokens/sec at seq_len={best_throughput_single['seq_len']}")
    print(f"Multi-Head: {best_throughput_multi['throughput_tokens_per_sec']:.0f} tokens/sec at seq_len={best_throughput_multi['seq_len']}")
    
    return {
        'scaling': scaling_df,
        'heads': heads_df,
        'embedding': embed_df,
        'memory': memory_df
    }


def main():
    """Run all performance analyses"""
    print("⚡ Self-Attention Performance Analysis")
    print("=" * 50)
    
    # Run comprehensive analysis
    results = comprehensive_performance_report()
    
    print("\n✅ All performance analyses completed!")
    
    return results


if __name__ == "__main__":
    main()
