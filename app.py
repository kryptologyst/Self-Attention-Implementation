"""
Interactive Web UI for Self-Attention Implementation
==================================================

A Streamlit-based web interface for exploring and visualizing self-attention mechanisms.
Features include:
- Interactive parameter adjustment
- Real-time attention visualization
- Performance benchmarking
- Multi-head attention exploration
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import pandas as pd
from typing import List, Tuple

# Import our attention implementation
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from attention_implementation import (
    SelfAttention, MultiHeadAttention, PositionalEncoding,
    AttentionConfig, AttentionVisualizer, AttentionBenchmark,
    create_causal_mask
)


def create_sample_text_embeddings(text: str, embed_dim: int = 64) -> Tuple[torch.Tensor, List[str]]:
    """Create simple embeddings from text tokens"""
    tokens = text.split()
    seq_len = len(tokens)
    
    # Create random embeddings (in practice, these would come from a trained model)
    embeddings = torch.randn(1, seq_len, embed_dim)
    
    return embeddings, tokens


def plot_attention_interactive(attn_weights: torch.Tensor, tokens: List[str], 
                             title: str = "Attention Weights") -> go.Figure:
    """Create interactive Plotly heatmap for attention weights"""
    attn_np = attn_weights.squeeze(0).detach().cpu().numpy()
    
    fig = go.Figure(data=go.Heatmap(
        z=attn_np,
        x=tokens,
        y=tokens,
        colorscale='Blues',
        hoverongaps=False,
        text=np.round(attn_np, 3),
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Key Position",
        yaxis_title="Query Position",
        width=600,
        height=500
    )
    
    return fig


def plot_multihead_attention_interactive(attn_weights: torch.Tensor, tokens: List[str]) -> go.Figure:
    """Create interactive multi-head attention visualization"""
    batch_size, num_heads, seq_len, _ = attn_weights.shape
    attn_np = attn_weights.squeeze(0).detach().cpu().numpy()
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=(num_heads + 1) // 2,
        subplot_titles=[f'Head {i+1}' for i in range(num_heads)],
        specs=[[{"type": "heatmap"} for _ in range((num_heads + 1) // 2)] for _ in range(2)]
    )
    
    for head in range(num_heads):
        row = head // ((num_heads + 1) // 2) + 1
        col = head % ((num_heads + 1) // 2) + 1
        
        fig.add_trace(
            go.Heatmap(
                z=attn_np[head],
                x=tokens,
                y=tokens,
                colorscale='Blues',
                showscale=False,
                text=np.round(attn_np[head], 2),
                texttemplate="%{text}",
                textfont={"size": 8}
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title="Multi-Head Attention Weights",
        height=800,
        showlegend=False
    )
    
    return fig


def benchmark_performance(embed_dim: int, num_heads: int, seq_len: int, 
                        batch_size: int = 1) -> dict:
    """Benchmark attention performance"""
    benchmark = AttentionBenchmark()
    
    results = {}
    
    # Single-head attention
    single_head = SelfAttention(embed_dim)
    single_result = benchmark.benchmark_attention(
        single_head, (batch_size, seq_len, embed_dim), num_runs=50, warmup_runs=5
    )
    results['Single-Head'] = single_result
    
    # Multi-head attention
    config = AttentionConfig(embed_dim=embed_dim, num_heads=num_heads)
    multi_head = MultiHeadAttention(config)
    multi_result = benchmark.benchmark_attention(
        multi_head, (batch_size, seq_len, embed_dim), num_runs=50, warmup_runs=5
    )
    results['Multi-Head'] = multi_result
    
    return results


def main():
    st.set_page_config(
        page_title="Self-Attention Explorer",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Self-Attention Implementation Explorer")
    st.markdown("Interactive exploration of self-attention mechanisms used in Transformers")
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Model parameters
    embed_dim = st.sidebar.slider("Embedding Dimension", 32, 512, 64, 32)
    num_heads = st.sidebar.slider("Number of Heads (Multi-Head)", 1, 16, 8, 1)
    seq_len = st.sidebar.slider("Sequence Length", 4, 32, 8, 1)
    dropout = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.1, 0.05)
    
    # Input text
    st.sidebar.header("📝 Input Text")
    sample_texts = [
        "The quick brown fox jumps over the lazy dog",
        "Attention is all you need for transformers",
        "Self attention allows each token to attend to all other tokens",
        "Machine learning models learn patterns from data"
    ]
    
    selected_text = st.sidebar.selectbox("Choose sample text:", sample_texts)
    custom_text = st.sidebar.text_input("Or enter custom text:", value="")
    
    if custom_text:
        input_text = custom_text
    else:
        input_text = selected_text
    
    # Truncate text to fit sequence length
    tokens = input_text.split()[:seq_len]
    input_text = " ".join(tokens)
    
    st.sidebar.write(f"**Tokens:** {tokens}")
    
    # Create embeddings
    embeddings, token_list = create_sample_text_embeddings(input_text, embed_dim)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Single-Head Attention", "🎯 Multi-Head Attention", "⚡ Performance", "📊 Analysis"])
    
    with tab1:
        st.header("Single-Head Self-Attention")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create single-head attention
            single_attn = SelfAttention(embed_dim, dropout)
            output, attn_weights = single_attn(embeddings)
            
            # Plot attention weights
            fig = plot_attention_interactive(attn_weights, token_list, "Single-Head Attention Weights")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Attention Statistics")
            
            attn_np = attn_weights.squeeze(0).detach().cpu().numpy()
            
            # Calculate statistics
            max_attention = np.max(attn_np)
            min_attention = np.min(attn_np)
            mean_attention = np.mean(attn_np)
            
            st.metric("Max Attention", f"{max_attention:.3f}")
            st.metric("Min Attention", f"{min_attention:.3f}")
            st.metric("Mean Attention", f"{mean_attention:.3f}")
            
            # Attention distribution
            st.subheader("📊 Attention Distribution")
            fig_dist = px.histogram(
                x=attn_np.flatten(),
                nbins=20,
                title="Attention Weight Distribution"
            )
            st.plotly_chart(fig_dist, use_container_width=True)
    
    with tab2:
        st.header("Multi-Head Self-Attention")
        
        if embed_dim % num_heads != 0:
            st.error(f"Embedding dimension ({embed_dim}) must be divisible by number of heads ({num_heads})")
            st.stop()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create multi-head attention
            config = AttentionConfig(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
            multi_attn = MultiHeadAttention(config)
            output, attn_weights = multi_attn(embeddings)
            
            # Plot multi-head attention
            fig = plot_multihead_attention_interactive(attn_weights, token_list)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Head Analysis")
            
            # Analyze each head
            attn_np = attn_weights.squeeze(0).detach().cpu().numpy()
            
            head_stats = []
            for head in range(num_heads):
                head_weights = attn_np[head]
                head_stats.append({
                    'Head': head + 1,
                    'Max Attention': np.max(head_weights),
                    'Min Attention': np.min(head_weights),
                    'Mean Attention': np.mean(head_weights),
                    'Std Attention': np.std(head_weights)
                })
            
            df_stats = pd.DataFrame(head_stats)
            st.dataframe(df_stats, use_container_width=True)
            
            # Head diversity
            st.subheader("🔄 Head Diversity")
            head_means = [np.mean(attn_np[head]) for head in range(num_heads)]
            fig_diversity = px.bar(
                x=list(range(1, num_heads + 1)),
                y=head_means,
                title="Mean Attention per Head",
                labels={'x': 'Head Number', 'y': 'Mean Attention'}
            )
            st.plotly_chart(fig_diversity, use_container_width=True)
    
    with tab3:
        st.header("⚡ Performance Benchmarking")
        
        if st.button("🚀 Run Benchmark", type="primary"):
            with st.spinner("Running performance benchmark..."):
                results = benchmark_performance(embed_dim, num_heads, seq_len)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("⏱️ Timing Results")
                
                timing_data = []
                for model_name, result in results.items():
                    timing_data.append({
                        'Model': model_name,
                        'Avg Time (ms)': result['avg_time_ms'],
                        'Throughput (tokens/sec)': result['throughput_tokens_per_sec']
                    })
                
                df_timing = pd.DataFrame(timing_data)
                st.dataframe(df_timing, use_container_width=True)
                
                # Timing comparison chart
                fig_timing = px.bar(
                    df_timing,
                    x='Model',
                    y='Avg Time (ms)',
                    title="Average Processing Time Comparison"
                )
                st.plotly_chart(fig_timing, use_container_width=True)
            
            with col2:
                st.subheader("📊 Throughput Results")
                
                # Throughput comparison chart
                fig_throughput = px.bar(
                    df_timing,
                    x='Model',
                    y='Throughput (tokens/sec)',
                    title="Throughput Comparison"
                )
                st.plotly_chart(fig_throughput, use_container_width=True)
                
                # Device info
                st.subheader("💻 Device Information")
                device_info = results['Single-Head']['device']
                st.info(f"Running on: {device_info}")
    
    with tab4:
        st.header("📊 Advanced Analysis")
        
        # Causal mask analysis
        st.subheader("🎭 Causal Mask Analysis")
        
        if st.checkbox("Apply Causal Mask"):
            mask = create_causal_mask(seq_len)
            
            # Single-head with mask
            single_attn = SelfAttention(embed_dim, dropout)
            output_masked, attn_weights_masked = single_attn(embeddings, mask)
            
            fig_masked = plot_attention_interactive(attn_weights_masked, token_list, "Causal Masked Attention")
            st.plotly_chart(fig_masked, use_container_width=True)
            
            st.info("🔒 Causal mask prevents attention to future tokens (upper triangular region)")
        
        # Positional encoding analysis
        st.subheader("📍 Positional Encoding Analysis")
        
        if st.checkbox("Show Positional Encoding"):
            pos_enc = PositionalEncoding(embed_dim)
            
            # Create dummy input for positional encoding
            dummy_input = torch.randn(seq_len, 1, embed_dim)
            pe_output = pos_enc(dummy_input)
            
            # Visualize positional encoding
            pe_matrix = pos_enc.pe[:seq_len, 0, :].numpy()
            
            fig_pe = go.Figure(data=go.Heatmap(
                z=pe_matrix.T,
                colorscale='RdYlBu',
                title="Positional Encoding Matrix"
            ))
            
            fig_pe.update_layout(
                xaxis_title="Position",
                yaxis_title="Embedding Dimension",
                height=400
            )
            
            st.plotly_chart(fig_pe, use_container_width=True)
        
        # Attention pattern analysis
        st.subheader("🔍 Attention Pattern Analysis")
        
        # Create attention without any modifications
        single_attn = SelfAttention(embed_dim, dropout)
        _, attn_weights = single_attn(embeddings)
        attn_np = attn_weights.squeeze(0).detach().cpu().numpy()
        
        # Analyze attention patterns
        col1, col2 = st.columns(2)
        
        with col1:
            # Attention entropy (diversity of attention)
            entropy = -np.sum(attn_np * np.log(attn_np + 1e-8), axis=1)
            fig_entropy = px.bar(
                x=token_list,
                y=entropy,
                title="Attention Entropy per Token",
                labels={'x': 'Token', 'y': 'Entropy'}
            )
            st.plotly_chart(fig_entropy, use_container_width=True)
        
        with col2:
            # Attention sparsity
            sparsity = np.sum(attn_np < 0.01, axis=1) / seq_len
            fig_sparsity = px.bar(
                x=token_list,
                y=sparsity,
                title="Attention Sparsity per Token",
                labels={'x': 'Token', 'y': 'Sparsity Ratio'}
            )
            st.plotly_chart(fig_sparsity, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 🧠 About Self-Attention
    
    Self-attention is the core mechanism in Transformer architectures that allows each token in a sequence 
    to attend to all other tokens, enabling the model to capture long-range dependencies and contextual relationships.
    
    **Key Concepts:**
    - **Query, Key, Value**: Each token generates Q, K, V vectors through learned linear transformations
    - **Attention Scores**: Computed as scaled dot-product between Q and K
    - **Attention Weights**: Softmax-normalized scores that sum to 1
    - **Context Vectors**: Weighted sum of Value vectors using attention weights
    - **Multi-Head**: Parallel attention heads capture different types of relationships
    """)
    
    st.markdown("Built with ❤️ using PyTorch, Streamlit, and Plotly")


if __name__ == "__main__":
    main()
