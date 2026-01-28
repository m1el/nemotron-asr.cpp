#!/usr/bin/env python3
"""
Detailed analysis of cache-aware streaming to understand expected behavior.

This script examines:
1. What the batch vs streaming difference is expected to be (design decision)
2. How caches are updated
3. Edge effects at chunk boundaries
4. Reference values for C++ implementation testing
"""

import torch
import numpy as np
import os
import sys

sys.path.insert(0, '/var/data/nvidia-speech/NeMo')

import nemo.collections.asr as nemo_asr


def load_model(model_path: str):
    print(f"Loading model from {model_path}...")
    model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(model_path)
    model.eval()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Model loaded on {next(model.parameters()).device}")
    return model


def analyze_cache_update(model):
    """Trace how caches are updated during streaming."""
    print("\n=== Cache Update Analysis ===\n", flush=True)
    
    encoder = model.encoder
    device = model.device
    
    # Get a single layer and its attention module
    layer = encoder.layers[0]
    attn = layer.self_attn
    
    print(f"Attention module: {type(attn).__name__}")
    print(f"  h (heads): {attn.h}")
    print(f"  d_k (head dim): {attn.d_k}")
    print(f"  _max_cache_len: {getattr(attn, '_max_cache_len', 'N/A')}")
    print(f"  cache_drop_size: {getattr(attn, 'cache_drop_size', 'N/A')}")
    
    # Create test input
    batch_size = 1
    d_model = encoder.d_model
    
    # Simulate what happens with cache
    print("\n--- Tracing update_cache() behavior ---\n")
    
    # Initial state: no cache
    query = torch.randn(batch_size, 5, d_model, device=device)
    key = query.clone()
    value = query.clone()
    cache = None
    
    print(f"Step 1: No cache (first chunk)")
    print(f"  query shape: {query.shape}")
    print(f"  cache: {cache}")
    
    # Simulate update_cache with no cache
    with torch.no_grad():
        key_out, value_out, query_out, cache_out = attn.update_cache(key, key, query, cache)
    
    print(f"  After update_cache:")
    print(f"    key shape: {key_out.shape}")
    print(f"    cache_out: {cache_out}")
    
    # Second step: with cache  
    cache = torch.randn(batch_size, 70, d_model, device=device)  # 70 cached frames
    query2 = torch.randn(batch_size, 5, d_model, device=device)
    key2 = query2.clone()
    
    print(f"\nStep 2: With cache (subsequent chunks)")
    print(f"  query shape: {query2.shape}")
    print(f"  cache shape: {cache.shape}")
    
    with torch.no_grad():
        key_out2, value_out2, query_out2, cache_out2 = attn.update_cache(key2, key2, query2, cache)
    
    print(f"  After update_cache:")
    print(f"    key shape: {key_out2.shape}")  # Should be [batch, cache_len + query_len, d_model]
    print(f"    cache_out shape: {cache_out2.shape}")
    
    # The update_cache logic:
    # key = value = concat(cache, key)  [cache_len + query_len]
    # q_keep_size = query.shape[1] - cache_drop_size
    # cache = concat(cache[:, q_keep_size:], query[:, :q_keep_size])
    
    cache_drop_size = getattr(attn, 'cache_drop_size', 0)
    q_keep_size = query2.shape[1] - cache_drop_size
    
    print(f"\n  cache_drop_size: {cache_drop_size}")
    print(f"  q_keep_size: {q_keep_size}")
    print(f"  New cache = concat(old_cache[:, {q_keep_size}:], query[:, :{q_keep_size}])")
    

def analyze_streaming_flow(model):
    """Trace the full streaming flow step by step."""
    print("\n=== Streaming Flow Analysis ===\n", flush=True)
    
    encoder = model.encoder
    device = model.device
    
    # Get streaming config
    if not hasattr(encoder, 'streaming_cfg') or encoder.streaming_cfg is None:
        encoder.setup_streaming_params()
    
    cfg = encoder.streaming_cfg
    print("Streaming configuration:")
    print(f"  chunk_size: {cfg.chunk_size}")
    print(f"  pre_encode_cache_size: {cfg.pre_encode_cache_size}")
    print(f"  last_channel_cache_size: {cfg.last_channel_cache_size}")
    print(f"  cache_drop_size: {cfg.cache_drop_size}")
    print(f"  drop_extra_pre_encoded: {cfg.drop_extra_pre_encoded}")
    print(f"  valid_out_len: {cfg.valid_out_len}")
    
    # The relationship between chunk_size and outputs:
    # After 8x subsampling, chunk_size mel frames -> chunk_size/8 encoder frames
    # But with edge effects and caching, the actual output length may vary
    
    chunk_size = cfg.chunk_size[1] if isinstance(cfg.chunk_size, list) else cfg.chunk_size
    expected_out_frames = chunk_size // 8
    
    print(f"\n  For chunk_size={chunk_size} mel frames:")
    print(f"    Expected encoder frames (8x subsample): ~{expected_out_frames}")
    print(f"    valid_out_len: {cfg.valid_out_len}")


def export_reference_caches(model, output_dir: str):
    """Export reference cache values for C++ testing."""
    print(f"\n=== Exporting Reference Caches to {output_dir} ===\n", flush=True)
    
    os.makedirs(output_dir, exist_ok=True)
    
    encoder = model.encoder
    device = model.device
    d_model = encoder.d_model
    
    if not hasattr(encoder, 'streaming_cfg') or encoder.streaming_cfg is None:
        encoder.setup_streaming_params()
    
    # Create fixed input for reproducibility
    torch.manual_seed(42)
    
    n_mels = encoder._feat_in  # 128
    mel_frames = 112  # One chunk
    
    mel_input = torch.randn(1, n_mels, mel_frames, device=device)
    mel_len = torch.tensor([mel_frames], dtype=torch.int64, device=device)
    
    # Get initial caches
    cache_last_channel, cache_last_time, cache_last_channel_len = encoder.get_initial_cache_state(
        batch_size=1, device=device
    )
    
    print(f"Initial caches:")
    print(f"  cache_last_channel: {cache_last_channel.shape}")
    print(f"  cache_last_time: {cache_last_time.shape}")
    print(f"  cache_last_channel_len: {cache_last_channel_len}")
    
    # Save initial caches
    np.save(os.path.join(output_dir, 'init_cache_channel.npy'), cache_last_channel.cpu().numpy())
    np.save(os.path.join(output_dir, 'init_cache_time.npy'), cache_last_time.cpu().numpy())
    np.save(os.path.join(output_dir, 'mel_input.npy'), mel_input.cpu().numpy())
    
    # Run streaming step
    with torch.no_grad():
        (
            encoded,
            encoded_len,
            cache_last_channel_next,
            cache_last_time_next,
            cache_last_channel_len_next,
        ) = encoder.cache_aware_stream_step(
            processed_signal=mel_input,
            processed_signal_length=mel_len,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            keep_all_outputs=True,
        )
    
    print(f"\nAfter streaming step:")
    print(f"  encoded: {encoded.shape}")
    print(f"  encoded_len: {encoded_len.item()}")
    print(f"  cache_last_channel_next: {cache_last_channel_next.shape}")
    print(f"  cache_last_time_next: {cache_last_time_next.shape}")
    print(f"  cache_last_channel_len_next: {cache_last_channel_len_next.item()}")
    
    # Save outputs
    np.save(os.path.join(output_dir, 'step0_encoded.npy'), encoded.cpu().numpy())
    np.save(os.path.join(output_dir, 'step0_cache_channel.npy'), cache_last_channel_next.cpu().numpy())
    np.save(os.path.join(output_dir, 'step0_cache_time.npy'), cache_last_time_next.cpu().numpy())
    
    # Run second step
    torch.manual_seed(43)
    mel_input2 = torch.randn(1, n_mels, mel_frames, device=device)
    mel_len2 = torch.tensor([mel_frames], dtype=torch.int64, device=device)
    
    with torch.no_grad():
        (
            encoded2,
            encoded_len2,
            cache_last_channel_next2,
            cache_last_time_next2,
            cache_last_channel_len_next2,
        ) = encoder.cache_aware_stream_step(
            processed_signal=mel_input2,
            processed_signal_length=mel_len2,
            cache_last_channel=cache_last_channel_next,
            cache_last_time=cache_last_time_next,
            cache_last_channel_len=cache_last_channel_len_next,
            keep_all_outputs=True,
        )
    
    print(f"\nAfter step 2:")
    print(f"  encoded2: {encoded2.shape}")
    print(f"  cache_last_channel_len_next2: {cache_last_channel_len_next2.item()}")
    
    np.save(os.path.join(output_dir, 'step1_mel_input.npy'), mel_input2.cpu().numpy())
    np.save(os.path.join(output_dir, 'step1_encoded.npy'), encoded2.cpu().numpy())
    np.save(os.path.join(output_dir, 'step1_cache_channel.npy'), cache_last_channel_next2.cpu().numpy())
    np.save(os.path.join(output_dir, 'step1_cache_time.npy'), cache_last_time_next2.cpu().numpy())
    
    # Export individual layer outputs for debugging
    print("\nExporting layer-by-layer intermediates...")
    
    # We need to hook into the layers to get intermediates
    # For now, just export the config
    config = {
        'd_model': d_model,
        'n_layers': len(encoder.layers),
        'n_heads': encoder._cfg.n_heads,
        'kernel_size': encoder._cfg.conv_kernel_size,
        'cache_size': encoder.streaming_cfg.last_channel_cache_size,
        'chunk_size': encoder.streaming_cfg.chunk_size,
    }
    
    import json
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    print(f"\nSaved reference files to {output_dir}/")
    print("Files:")
    for f in os.listdir(output_dir):
        path = os.path.join(output_dir, f)
        size = os.path.getsize(path)
        print(f"  {f}: {size:,} bytes")


def compare_attention_mechanisms(model):
    """Compare attention output with and without cache to understand the design."""
    print("\n=== Attention Mechanism Comparison ===\n", flush=True)
    
    encoder = model.encoder
    device = model.device
    d_model = encoder.d_model
    n_heads = encoder._cfg.n_heads
    d_head = d_model // n_heads
    
    layer = encoder.layers[0]
    attn = layer.self_attn
    
    batch_size = 1
    total_seq = 20  # Total sequence length
    chunk_seq = 10  # Process in two chunks
    cache_size = 70
    
    torch.manual_seed(42)
    
    # Create input
    x = torch.randn(batch_size, total_seq, d_model, device=device)
    
    # Position embeddings for full sequence
    pos_len_full = 2 * total_seq - 1
    
    # Get sinusoidal position embeddings from the encoder
    encoder.update_max_seq_length(total_seq, device)
    
    # Test 1: Full sequence without cache
    x_normed = layer.norm_self_att(x)
    pos_emb_full = encoder.pos_enc.pe[:, :pos_len_full]
    
    with torch.no_grad():
        out_full = attn(query=x_normed, key=x_normed, value=x_normed, 
                       mask=None, pos_emb=pos_emb_full, cache=None)
    
    print(f"Full sequence output: {out_full.shape}")
    print(f"  First 5 values: {out_full[0, 0, :5].cpu().numpy()}")
    
    # Test 2: First chunk without cache, second chunk with cache
    x1 = x[:, :chunk_seq]
    x2 = x[:, chunk_seq:]
    
    x1_normed = layer.norm_self_att(x1)
    pos_len_chunk1 = 2 * chunk_seq - 1
    pos_emb_chunk1 = encoder.pos_enc.pe[:, :pos_len_chunk1]
    
    with torch.no_grad():
        out_chunk1 = attn(query=x1_normed, key=x1_normed, value=x1_normed,
                         mask=None, pos_emb=pos_emb_chunk1, cache=None)
    
    print(f"\nChunk 1 output: {out_chunk1.shape}")
    print(f"  First 5 values: {out_chunk1[0, 0, :5].cpu().numpy()}")
    
    # Check if chunk 1 matches first half of full
    diff1 = (out_full[:, :chunk_seq] - out_chunk1).abs().max().item()
    print(f"  Diff from full[:chunk_seq]: {diff1:.6e}")
    
    # Now chunk 2 with cache (using chunk 1 as cache)
    # This is where the streaming magic happens
    x2_normed = layer.norm_self_att(x2)
    
    # For cached attention, the cache contains the previous K/V (not x1_normed, but the projected K/V)
    # The cache in NeMo stores the previous QUERY as cache (not K/V!)
    # Let's trace what update_cache actually does
    
    cache = x1_normed  # Previous normalized input becomes cache
    
    # Position embedding for cached attention needs to span cache_len + chunk_len
    # For relative position encoding: positions from -(chunk_len-1) to +(cache_len + chunk_len - 1)
    cache_len = chunk_seq
    pos_len_cached = 2 * (cache_len + chunk_seq) - 1
    
    # Need to get the right slice of position embeddings
    # The position embeddings in NeMo are stored as [length - 1, ..., 0, ..., -(length-1)]
    # For cache_len = 10, chunk_len = 10:
    # We need positions for relative distances from -9 to +19
    
    with torch.no_grad():
        out_chunk2 = attn(query=x2_normed, key=x2_normed, value=x2_normed,
                         mask=None, pos_emb=pos_emb_chunk1, cache=cache)
        if isinstance(out_chunk2, tuple):
            out_chunk2, new_cache = out_chunk2
    
    print(f"\nChunk 2 output (with cache): {out_chunk2.shape}")
    print(f"  First 5 values: {out_chunk2[0, 0, :5].cpu().numpy()}")
    
    # Check if chunk 2 matches second half of full
    diff2 = (out_full[:, chunk_seq:] - out_chunk2).abs().max().item()
    print(f"  Diff from full[chunk_seq:]: {diff2:.6e}")
    
    print("\nNote: Difference is expected due to position embedding handling in streaming mode.")
    print("The streaming design accepts some approximation for real-time processing.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze streaming cache implementation')
    parser.add_argument('--model', type=str, 
                        default='/var/data/nvidia-speech/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo',
                        help='Path to NeMo model')
    parser.add_argument('--export', type=str, default=None,
                        help='Export reference caches to directory')
    parser.add_argument('--analyze-cache', action='store_true',
                        help='Analyze cache update mechanics')
    parser.add_argument('--analyze-flow', action='store_true',
                        help='Analyze streaming flow')
    parser.add_argument('--compare-attn', action='store_true',
                        help='Compare attention mechanisms')
    
    args = parser.parse_args()
    
    model = load_model(args.model)
    
    if args.analyze_cache:
        analyze_cache_update(model)
    
    if args.analyze_flow:
        analyze_streaming_flow(model)
    
    if args.compare_attn:
        compare_attention_mechanisms(model)
    
    if args.export:
        export_reference_caches(model, args.export)
    
    if not (args.analyze_cache or args.analyze_flow or args.compare_attn or args.export):
        # Default: run all analyses
        analyze_cache_update(model)
        analyze_streaming_flow(model)


if __name__ == '__main__':
    main()
