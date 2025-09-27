# Advanced Self-Attention Implementation

A comprehensive implementation of self-attention mechanisms used in Transformer architectures, featuring modern PyTorch best practices, interactive visualization, and performance benchmarking.

## Features

- **Single-Head Self-Attention**: Core attention mechanism implementation
- **Multi-Head Self-Attention**: Parallel attention heads for diverse relationship capture
- **Positional Encoding**: Sinusoidal positional encoding for sequence position information
- **Interactive Web UI**: Streamlit-based interface for real-time exploration
- **Attention Visualization**: Heatmaps and interactive plots for attention patterns
- **Performance Benchmarking**: Comprehensive performance analysis tools
- **Comprehensive Testing**: Full test suite with pytest
- **Modern PyTorch**: Latest PyTorch best practices and optimizations

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/self-attention-implementation.git
cd self-attention-implementation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Basic Usage

```python
import torch
from attention_implementation import SelfAttention, MultiHeadAttention, AttentionConfig

# Single-head self-attention
embed_dim = 64
attention = SelfAttention(embed_dim)

# Create input tensor (batch_size, seq_len, embed_dim)
x = torch.randn(1, 8, 64)
output, attn_weights = attention(x)

print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attn_weights.shape}")
```

### Multi-Head Attention

```python
# Multi-head self-attention
config = AttentionConfig(embed_dim=64, num_heads=8)
multi_attention = MultiHeadAttention(config)

output, attn_weights = multi_attention(x)
print(f"Multi-head output shape: {output.shape}")
print(f"Multi-head attention weights shape: {attn_weights.shape}")
```

## Interactive Web Interface

Launch the interactive Streamlit app:

```bash
streamlit run app.py
```

The web interface provides:
- **Real-time parameter adjustment**
- **Interactive attention visualization**
- **Performance benchmarking**
- **Multi-head attention exploration**
- **Causal mask analysis**
- **Positional encoding visualization**

## Visualization Examples

### Single-Head Attention Heatmap
```python
from attention_implementation import AttentionVisualizer

visualizer = AttentionVisualizer()
visualizer.plot_attention_weights(attn_weights, tokens=["The", "quick", "brown", "fox"])
```

### Multi-Head Attention Visualization
```python
visualizer.plot_multihead_attention(attn_weights, tokens)
```

## Performance Benchmarking

```python
from attention_implementation import AttentionBenchmark

benchmark = AttentionBenchmark()
results = benchmark.benchmark_attention(model, input_shape=(1, 64, 64))
print(f"Average time: {results['avg_time_ms']:.2f} ms")
print(f"Throughput: {results['throughput_tokens_per_sec']:.0f} tokens/sec")
```

## Testing

Run the comprehensive test suite:

```bash
pytest test_attention.py -v
```

The test suite covers:
- Single-head and multi-head attention
- Positional encoding
- Attention visualization
- Performance benchmarking
- Integration tests

## 📁 Project Structure

```
self-attention-implementation/
├── attention_implementation.py    # Main implementation
├── app.py                        # Streamlit web interface
├── test_attention.py             # Comprehensive test suite
├── requirements.txt              # Dependencies
├── README.md                     # This file
└── examples/                     # Usage examples
    ├── basic_usage.py
    ├── visualization_demo.py
    └── performance_analysis.py
```

## 🔧 Configuration

The `AttentionConfig` dataclass allows easy configuration:

```python
@dataclass
class AttentionConfig:
    embed_dim: int = 512          # Embedding dimension
    num_heads: int = 8            # Number of attention heads
    dropout: float = 0.1          # Dropout rate
    max_seq_len: int = 1024       # Maximum sequence length
    bias: bool = True             # Use bias in linear layers
```

## Key Components

### SelfAttention Class
- Implements scaled dot-product attention
- Supports attention masking
- Configurable dropout and bias
- Returns attention weights for visualization

### MultiHeadAttention Class
- Parallel attention heads
- Efficient tensor reshaping
- Head concatenation and projection
- Multi-head attention weight visualization

### PositionalEncoding Class
- Sinusoidal positional encoding
- Configurable maximum sequence length
- Efficient buffer registration

### AttentionVisualizer Class
- Static methods for plotting
- Support for single-head and multi-head visualization
- Customizable styling and annotations

### AttentionBenchmark Class
- Performance measurement tools
- GPU/CPU automatic detection
- Throughput calculation
- Comprehensive timing analysis

## Understanding Self-Attention

Self-attention is the core mechanism that allows each token in a sequence to attend to all other tokens, enabling the model to capture contextual relationships and long-range dependencies.

### Mathematical Formulation

For input sequence **X** ∈ ℝ^(L×d), self-attention computes:

1. **Query, Key, Value projections**:
   - Q = XW_Q, K = XW_K, V = XW_V

2. **Attention scores**:
   - S = QK^T / √d

3. **Attention weights**:
   - A = softmax(S)

4. **Output**:
   - O = AV

### Multi-Head Extension

Multi-head attention runs multiple attention mechanisms in parallel:
- Each head has its own Q, K, V projections
- Outputs are concatenated and projected
- Allows capturing different types of relationships

## Advanced Features

### Causal Masking
```python
from attention_implementation import create_causal_mask

mask = create_causal_mask(seq_len=8)
output, attn_weights = attention(x, mask=mask)
```

### Custom Attention Patterns
The implementation supports custom attention masks for specialized attention patterns.

### Performance Optimization
- Efficient tensor operations
- GPU acceleration support
- Memory-optimized implementations
- Batch processing capabilities

## Performance Characteristics

| Model Type | Sequence Length | Embedding Dim | Avg Time (ms) | Throughput (tokens/sec) |
|------------|----------------|---------------|---------------|-------------------------|
| Single-Head | 64 | 64 | ~2.5 | ~25,600 |
| Multi-Head (8) | 64 | 64 | ~3.2 | ~20,000 |
| Multi-Head (16) | 64 | 64 | ~4.1 | ~15,600 |

*Benchmarks run on modern GPU hardware*

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Install development dependencies: `pip install -r requirements.txt`
4. Run tests: `pytest test_attention.py`
5. Commit changes: `git commit -m "Add feature"`
6. Push to branch: `git push origin feature-name`
7. Submit a Pull Request

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Visual explanation
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) - PyTorch reference

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The Transformer architecture by Vaswani et al.
- PyTorch team for the excellent deep learning framework
- Streamlit team for the interactive web framework
- The open-source community for inspiration and feedback

## Support

If you have any questions or need help, please:
- Open an issue on GitHub
- Check the documentation
- Review the test cases for usage examples


# Self-Attention-Implementation
