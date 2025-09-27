"""
Unit tests for Self-Attention Implementation
===========================================

Comprehensive test suite covering:
- Single-head self-attention
- Multi-head self-attention  
- Positional encoding
- Attention visualization
- Performance benchmarking
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from attention_implementation import (
    SelfAttention, MultiHeadAttention, PositionalEncoding,
    AttentionConfig, AttentionVisualizer, AttentionBenchmark,
    create_causal_mask
)


class TestSelfAttention:
    """Test cases for single-head self-attention"""
    
    def test_initialization(self):
        """Test SelfAttention initialization"""
        embed_dim = 64
        dropout = 0.1
        attention = SelfAttention(embed_dim, dropout)
        
        assert attention.embed_dim == embed_dim
        assert attention.scale == np.sqrt(embed_dim)
        assert isinstance(attention.query, nn.Linear)
        assert isinstance(attention.key, nn.Linear)
        assert isinstance(attention.value, nn.Linear)
        assert isinstance(attention.out_proj, nn.Linear)
        assert isinstance(attention.dropout, nn.Dropout)
    
    def test_forward_pass(self):
        """Test forward pass of self-attention"""
        batch_size, seq_len, embed_dim = 2, 8, 64
        attention = SelfAttention(embed_dim)
        
        x = torch.randn(batch_size, seq_len, embed_dim)
        output, attn_weights = attention(x)
        
        # Check output shapes
        assert output.shape == x.shape
        assert attn_weights.shape == (batch_size, seq_len, seq_len)
        
        # Check attention weights sum to 1
        assert torch.allclose(attn_weights.sum(dim=-1), torch.ones(batch_size, seq_len), atol=1e-6)
        
        # Check attention weights are non-negative
        assert torch.all(attn_weights >= 0)
    
    def test_with_mask(self):
        """Test self-attention with causal mask"""
        batch_size, seq_len, embed_dim = 1, 4, 32
        attention = SelfAttention(embed_dim)
        
        x = torch.randn(batch_size, seq_len, embed_dim)
        mask = create_causal_mask(seq_len)
        
        output, attn_weights = attention(x, mask)
        
        # Check that masked positions have zero attention
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert attn_weights[0, i, j] == 0
    
    def test_gradient_flow(self):
        """Test that gradients flow properly"""
        embed_dim = 32
        attention = SelfAttention(embed_dim)
        
        x = torch.randn(1, 4, embed_dim, requires_grad=True)
        output, _ = attention(x)
        
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestMultiHeadAttention:
    """Test cases for multi-head self-attention"""
    
    def test_initialization(self):
        """Test MultiHeadAttention initialization"""
        config = AttentionConfig(embed_dim=64, num_heads=8)
        attention = MultiHeadAttention(config)
        
        assert attention.embed_dim == config.embed_dim
        assert attention.num_heads == config.num_heads
        assert attention.head_dim == config.embed_dim // config.num_heads
        assert attention.scale == np.sqrt(attention.head_dim)
    
    def test_forward_pass(self):
        """Test forward pass of multi-head attention"""
        config = AttentionConfig(embed_dim=64, num_heads=8)
        attention = MultiHeadAttention(config)
        
        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, config.embed_dim)
        output, attn_weights = attention(x)
        
        # Check output shapes
        assert output.shape == x.shape
        assert attn_weights.shape == (batch_size, config.num_heads, seq_len, seq_len)
        
        # Check attention weights sum to 1 for each head
        for head in range(config.num_heads):
            head_weights = attn_weights[:, head, :, :]
            assert torch.allclose(head_weights.sum(dim=-1), torch.ones(batch_size, seq_len), atol=1e-6)
    
    def test_head_dimension_consistency(self):
        """Test that head dimensions are consistent"""
        config = AttentionConfig(embed_dim=64, num_heads=8)
        attention = MultiHeadAttention(config)
        
        assert config.embed_dim % config.num_heads == 0
        assert attention.head_dim == 8  # 64 / 8 = 8
    
    def test_invalid_head_configuration(self):
        """Test that invalid head configuration raises error"""
        config = AttentionConfig(embed_dim=64, num_heads=7)  # 64 % 7 != 0
        
        with pytest.raises(AssertionError):
            MultiHeadAttention(config)


class TestPositionalEncoding:
    """Test cases for positional encoding"""
    
    def test_initialization(self):
        """Test PositionalEncoding initialization"""
        embed_dim, max_seq_len = 64, 1000
        pos_enc = PositionalEncoding(embed_dim, max_seq_len)
        
        assert pos_enc.embed_dim == embed_dim
        assert pos_enc.pe.shape == (max_seq_len, 1, embed_dim)
    
    def test_forward_pass(self):
        """Test forward pass of positional encoding"""
        embed_dim, seq_len = 64, 16
        pos_enc = PositionalEncoding(embed_dim)
        
        x = torch.randn(seq_len, 1, embed_dim)
        output = pos_enc(x)
        
        assert output.shape == x.shape
        assert not torch.equal(output, x)  # Should be different due to positional encoding
    
    def test_positional_encoding_properties(self):
        """Test properties of positional encoding"""
        embed_dim = 64
        pos_enc = PositionalEncoding(embed_dim)
        
        pe_matrix = pos_enc.pe.squeeze(1).numpy()  # Remove batch dimension
        
        # Check that encoding values are bounded
        assert np.all(np.abs(pe_matrix) <= 1)
        
        # Check sinusoidal pattern for even dimensions
        even_dims = pe_matrix[:, ::2]
        odd_dims = pe_matrix[:, 1::2]
        
        # Even dimensions should follow sine pattern
        assert np.allclose(even_dims[0, :], np.zeros(even_dims.shape[1]))
        
        # Odd dimensions should follow cosine pattern  
        assert np.allclose(odd_dims[0, :], np.ones(odd_dims.shape[1]))


class TestAttentionConfig:
    """Test cases for AttentionConfig dataclass"""
    
    def test_default_values(self):
        """Test default configuration values"""
        config = AttentionConfig()
        
        assert config.embed_dim == 512
        assert config.num_heads == 8
        assert config.dropout == 0.1
        assert config.max_seq_len == 1024
        assert config.bias == True
    
    def test_custom_values(self):
        """Test custom configuration values"""
        config = AttentionConfig(
            embed_dim=256,
            num_heads=4,
            dropout=0.2,
            max_seq_len=512,
            bias=False
        )
        
        assert config.embed_dim == 256
        assert config.num_heads == 4
        assert config.dropout == 0.2
        assert config.max_seq_len == 512
        assert config.bias == False


class TestAttentionVisualizer:
    """Test cases for attention visualization"""
    
    def test_plot_attention_weights(self):
        """Test attention weight plotting"""
        visualizer = AttentionVisualizer()
        
        # Create mock attention weights
        attn_weights = torch.randn(1, 4, 4)
        tokens = ['token_0', 'token_1', 'token_2', 'token_3']
        
        # Mock matplotlib to avoid actual plotting during tests
        with patch('matplotlib.pyplot.show'):
            with patch('matplotlib.pyplot.figure'):
                with patch('seaborn.heatmap'):
                    visualizer.plot_attention_weights(attn_weights, tokens)
    
    def test_plot_multihead_attention(self):
        """Test multi-head attention plotting"""
        visualizer = AttentionVisualizer()
        
        # Create mock multi-head attention weights
        attn_weights = torch.randn(1, 8, 4, 4)  # batch, heads, seq, seq
        tokens = ['token_0', 'token_1', 'token_2', 'token_3']
        
        # Mock matplotlib to avoid actual plotting during tests
        with patch('matplotlib.pyplot.show'):
            with patch('matplotlib.pyplot.subplots'):
                with patch('seaborn.heatmap'):
                    visualizer.plot_multihead_attention(attn_weights, tokens)


class TestAttentionBenchmark:
    """Test cases for attention benchmarking"""
    
    def test_benchmark_attention(self):
        """Test attention benchmarking"""
        benchmark = AttentionBenchmark()
        
        # Create a simple model
        model = SelfAttention(64)
        input_shape = (1, 8, 64)
        
        result = benchmark.benchmark_attention(model, input_shape, num_runs=5, warmup_runs=2)
        
        assert 'avg_time_ms' in result
        assert 'throughput_tokens_per_sec' in result
        assert 'device' in result
        assert 'input_shape' in result
        
        assert result['avg_time_ms'] > 0
        assert result['throughput_tokens_per_sec'] > 0
        assert result['input_shape'] == input_shape


class TestUtilityFunctions:
    """Test cases for utility functions"""
    
    def test_create_causal_mask(self):
        """Test causal mask creation"""
        seq_len = 4
        mask = create_causal_mask(seq_len)
        
        assert mask.shape == (1, seq_len, seq_len)
        
        # Check that upper triangular part is zero
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert mask[0, i, j] == 0
        
        # Check that lower triangular part (including diagonal) is one
        for i in range(seq_len):
            for j in range(i + 1):
                assert mask[0, i, j] == 1


class TestIntegration:
    """Integration tests"""
    
    def test_end_to_end_single_head(self):
        """Test complete single-head attention pipeline"""
        embed_dim = 32
        attention = SelfAttention(embed_dim)
        
        # Create input with positional encoding
        seq_len, batch_size = 8, 1
        x = torch.randn(batch_size, seq_len, embed_dim)
        
        # Apply attention
        output, attn_weights = attention(x)
        
        # Verify outputs
        assert output.shape == x.shape
        assert attn_weights.shape == (batch_size, seq_len, seq_len)
        
        # Verify attention properties
        assert torch.allclose(attn_weights.sum(dim=-1), torch.ones(batch_size, seq_len), atol=1e-6)
        assert torch.all(attn_weights >= 0)
    
    def test_end_to_end_multi_head(self):
        """Test complete multi-head attention pipeline"""
        config = AttentionConfig(embed_dim=64, num_heads=8)
        attention = MultiHeadAttention(config)
        
        # Create input
        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, config.embed_dim)
        
        # Apply attention
        output, attn_weights = attention(x)
        
        # Verify outputs
        assert output.shape == x.shape
        assert attn_weights.shape == (batch_size, config.num_heads, seq_len, seq_len)
        
        # Verify attention properties for each head
        for head in range(config.num_heads):
            head_weights = attn_weights[:, head, :, :]
            assert torch.allclose(head_weights.sum(dim=-1), torch.ones(batch_size, seq_len), atol=1e-6)
            assert torch.all(head_weights >= 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
