#!/usr/bin/env python3
"""
Export detailed layer-by-layer outputs from NeMo for C++ validation.

This exports:
1. Subsampling input/output
2. Each conformer layer output (with and without cache)
3. Full encoder output
4. Cache state after each step
"""

import torch
import numpy as np
import os
import sys
import json

sys.path.insert(0, '/var/data/nvidia-speech/NeMo')

import nemo.collections.asr as nemo_asr


def load_model(model_path: str):
    print(f"Loading model from {model_path}...")
    model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(model_path)
    model.eval()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Model loaded on {next(model.parameters()).device}")
    return model


def export_subsampling_test(model, output_dir: str):
    """Export subsampling layer input/output for testing."""
    print("\n=== Exporting Subsampling Test Data ===", flush=True)
    
    encoder = model.encoder
    device = model.device
    
    torch.manual_seed(42)
    
    n_mels = encoder._feat_in  # 128
    n_frames = 112  # One chunk
    
    mel_input = torch.randn(1, n_mels, n_frames, device=device)
    mel_len = torch.tensor([n_frames], dtype=torch.int64, device=device)
    
    # Get subsampling output
    subsampling = encoder.pre_encode
    
    with torch.no_grad():
        # Apply preprocessing (subsampling)
        audio_signal, length = subsampling(mel_input.transpose(1, 2), mel_len)
    
    print(f"Subsampling input: {mel_input.shape}")  # [1, 128, 112]
    print(f"Subsampling output: {audio_signal.shape}")  # [1, T, 1024]
    print(f"Output length: {length.item()}")
    
    np.save(os.path.join(output_dir, 'sub_input.npy'), mel_input.cpu().numpy())
    np.save(os.path.join(output_dir, 'sub_output.npy'), audio_signal.cpu().numpy())
    np.save(os.path.join(output_dir, 'sub_output_len.npy'), length.cpu().numpy())


def hook_conformer_layer(model, layer_idx: int, output_dir: str):
    """Add hooks to capture layer inputs/outputs."""
    
    outputs = {}
    
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                outputs[name] = output[0].detach().cpu()
            else:
                outputs[name] = output.detach().cpu()
        return hook
    
    layer = model.encoder.layers[layer_idx]
    
    # Hook the main components
    hooks = []
    hooks.append(layer.norm_feed_forward1.register_forward_hook(make_hook('ffn1_ln')))
    hooks.append(layer.feed_forward1.register_forward_hook(make_hook('ffn1')))
    hooks.append(layer.norm_self_att.register_forward_hook(make_hook('attn_ln')))
    hooks.append(layer.self_attn.register_forward_hook(make_hook('attn')))
    hooks.append(layer.norm_conv.register_forward_hook(make_hook('conv_ln')))
    hooks.append(layer.conv.register_forward_hook(make_hook('conv')))  # Note: 'conv' not 'conv_module'
    hooks.append(layer.norm_feed_forward2.register_forward_hook(make_hook('ffn2_ln')))
    hooks.append(layer.feed_forward2.register_forward_hook(make_hook('ffn2')))
    hooks.append(layer.norm_out.register_forward_hook(make_hook('final_ln')))
    
    return hooks, outputs


def export_layer_test(model, output_dir: str, layer_idx: int = 0):
    """Export single layer input/output for testing."""
    print(f"\n=== Exporting Layer {layer_idx} Test Data ===", flush=True)
    
    encoder = model.encoder
    device = model.device
    
    torch.manual_seed(42)
    
    n_mels = encoder._feat_in  # 128
    n_frames = 112  # One chunk
    
    mel_input = torch.randn(1, n_mels, n_frames, device=device)
    mel_len = torch.tensor([n_frames], dtype=torch.int64, device=device)
    
    # Add hooks to capture layer outputs
    hooks, outputs = hook_conformer_layer(model, layer_idx, output_dir)
    
    try:
        with torch.no_grad():
            # Run full encoder (batch mode, no streaming)
            encoded, encoded_len = encoder(audio_signal=mel_input, length=mel_len)
    finally:
        # Remove hooks
        for h in hooks:
            h.remove()
    
    print(f"Layer {layer_idx} outputs captured:")
    for name, tensor in outputs.items():
        print(f"  {name}: {tensor.shape}")
        np.save(os.path.join(output_dir, f'layer{layer_idx}_{name}.npy'), tensor.numpy())


def export_streaming_step(model, output_dir: str):
    """Export one streaming step with cache update for testing."""
    print("\n=== Exporting Streaming Step Data ===", flush=True)
    
    encoder = model.encoder
    device = model.device
    
    if not hasattr(encoder, 'streaming_cfg') or encoder.streaming_cfg is None:
        encoder.setup_streaming_params()
    
    torch.manual_seed(42)
    
    n_mels = encoder._feat_in  # 128
    n_frames = 112  # One chunk
    
    mel_input = torch.randn(1, n_mels, n_frames, device=device)
    mel_len = torch.tensor([n_frames], dtype=torch.int64, device=device)
    
    # Get initial caches
    cache_last_channel, cache_last_time, cache_last_channel_len = encoder.get_initial_cache_state(
        batch_size=1, device=device
    )
    
    print(f"Initial caches:")
    print(f"  cache_last_channel: {cache_last_channel.shape}")  # [24, 1, 70, 1024]
    print(f"  cache_last_time: {cache_last_time.shape}")  # [24, 1, 1024, 8]
    
    # Save initial caches
    np.save(os.path.join(output_dir, 'stream_init_cache_channel.npy'), cache_last_channel.cpu().numpy())
    np.save(os.path.join(output_dir, 'stream_init_cache_time.npy'), cache_last_time.cpu().numpy())
    np.save(os.path.join(output_dir, 'stream_mel_input.npy'), mel_input.cpu().numpy())
    
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
    print(f"  encoded: {encoded.shape}")  # [1, 1024, T]
    print(f"  encoded_len: {encoded_len.item()}")
    
    # Save outputs
    np.save(os.path.join(output_dir, 'stream_encoded.npy'), encoded.cpu().numpy())
    np.save(os.path.join(output_dir, 'stream_encoded_len.npy'), np.array([encoded_len.item()]))
    np.save(os.path.join(output_dir, 'stream_cache_channel_next.npy'), cache_last_channel_next.cpu().numpy())
    np.save(os.path.join(output_dir, 'stream_cache_time_next.npy'), cache_last_time_next.cpu().numpy())
    
    # Run batch mode with same input for comparison
    with torch.no_grad():
        encoded_batch, encoded_len_batch = encoder(audio_signal=mel_input, length=mel_len)
    
    print(f"\nBatch mode comparison:")
    print(f"  encoded_batch: {encoded_batch.shape}")
    
    np.save(os.path.join(output_dir, 'batch_encoded.npy'), encoded_batch.cpu().numpy())


def export_pos_encoding(model, output_dir: str):
    """Export positional encoding for testing."""
    print("\n=== Exporting Positional Encoding ===", flush=True)
    
    encoder = model.encoder
    
    # Get positional encoding
    pos_emb = encoder.pos_enc.pe
    print(f"Positional encoding shape: {pos_emb.shape}")  # [1, 2*max_len-1, d_model]
    
    # Just save first 200 positions for testing
    np.save(os.path.join(output_dir, 'pos_emb.npy'), pos_emb[:, :500].cpu().numpy())


def export_attention_test(model, output_dir: str):
    """Export attention layer with intermediate values for testing."""
    print("\n=== Exporting Attention Test Data ===", flush=True)
    
    encoder = model.encoder
    device = model.device
    
    torch.manual_seed(42)
    
    d_model = encoder.d_model
    n_heads = encoder._cfg.n_heads
    d_head = d_model // n_heads
    
    layer = encoder.layers[0]
    attn = layer.self_attn
    
    # Create test input
    seq_len = 13  # Typical chunk after subsampling
    x = torch.randn(1, seq_len, d_model, device=device)
    
    # Get position embeddings
    encoder.update_max_seq_length(seq_len, device)
    pos_len = 2 * seq_len - 1
    pos_emb = encoder.pos_enc.pe[:, :pos_len]
    
    print(f"Attention test input: {x.shape}")
    print(f"Position embedding: {pos_emb.shape}")
    
    with torch.no_grad():
        out = attn(query=x, key=x, value=x, mask=None, pos_emb=pos_emb, cache=None)
    
    print(f"Attention output: {out.shape}")
    
    np.save(os.path.join(output_dir, 'attn_input.npy'), x.cpu().numpy())
    np.save(os.path.join(output_dir, 'attn_pos_emb.npy'), pos_emb.cpu().numpy())
    np.save(os.path.join(output_dir, 'attn_output.npy'), out.cpu().numpy())
    
    # Also export Q, K, V weights for verification
    np.save(os.path.join(output_dir, 'attn_q_w.npy'), attn.linear_q.weight.detach().cpu().numpy())
    np.save(os.path.join(output_dir, 'attn_k_w.npy'), attn.linear_k.weight.detach().cpu().numpy())
    np.save(os.path.join(output_dir, 'attn_v_w.npy'), attn.linear_v.weight.detach().cpu().numpy())
    np.save(os.path.join(output_dir, 'attn_out_w.npy'), attn.linear_out.weight.detach().cpu().numpy())
    np.save(os.path.join(output_dir, 'attn_pos_w.npy'), attn.linear_pos.weight.detach().cpu().numpy())
    np.save(os.path.join(output_dir, 'attn_pos_bias_u.npy'), attn.pos_bias_u.detach().cpu().numpy())
    np.save(os.path.join(output_dir, 'attn_pos_bias_v.npy'), attn.pos_bias_v.detach().cpu().numpy())


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Export layer data for C++ testing')
    parser.add_argument('--model', type=str, 
                        default='/var/data/nvidia-speech/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo',
                        help='Path to NeMo model')
    parser.add_argument('--output', type=str, default='scripts/layer_outputs',
                        help='Output directory')
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    model = load_model(args.model)
    
    # Export all test data
    export_subsampling_test(model, args.output)
    export_pos_encoding(model, args.output)
    export_attention_test(model, args.output)
    export_layer_test(model, args.output, layer_idx=0)
    export_streaming_step(model, args.output)
    
    print(f"\n=== Saved all exports to {args.output}/ ===")
    for f in sorted(os.listdir(args.output)):
        path = os.path.join(args.output, f)
        size = os.path.getsize(path)
        print(f"  {f}: {size:,} bytes")


if __name__ == '__main__':
    main()
