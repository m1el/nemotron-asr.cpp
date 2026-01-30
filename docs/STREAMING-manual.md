Example streaming implemented in `scripts/my_streaming.py`.
Run using `uv run scripts/my_streaming.py`.

Model config audio:
sample_rate: 16000
window_size: 0.025sec / 400samples
window_stride: 0.01sec / 160samples

Audio chunking is defined by `att_context_size = (left_chunks_num, chunk_size)`

`cache_last_channel, cache_last_time, cache_last_channel_len`
is the streaming state preserved between the runs.

mel chunking: 121

The sizes of these tensors are defined by
n_layers: 24
d_model: 1024
conv_kernel_size: 9
conv_context_size: 'causal' -> [(conv_kernel_size-1), 0] -> [8, 0]

streaming_cfg(70, 0) = (
    chunk_size=[1, 8],
    shift_size=[1, 8],
    cache_drop_size=0,
    last_channel_cache_size=70,
    valid_out_len=1,
    pre_encode_cache_size=[0, 9],
    drop_extra_pre_encoded=2,
    last_channel_num=0,
    last_time_num=0,
)
streaming_cfg(70, 1) = (
    chunk_size=[9, 16],
    shift_size=[9, 16],
    cache_drop_size=0,
    last_channel_cache_size=70,
    valid_out_len=2,
    pre_encode_cache_size=[0, 9],
    drop_extra_pre_encoded=2,
    last_channel_num=0,
    last_time_num=0
)
streaming_cfg(70, 6) = (
    chunk_size=[49, 56],
    shift_size=[49, 56],
    cache_drop_size=0,
    last_channel_cache_size=70,
    valid_out_len=7,
    pre_encode_cache_size=[0, 9],
    drop_extra_pre_encoded=2,
    last_channel_num=0,
    last_time_num=0
)
streaming_cfg(70, 6) = (
    chunk_size=[105, 112],
    shift_size=[105, 112],
    cache_drop_size=0,
    last_channel_cache_size=70,
    valid_out_len=14,
    pre_encode_cache_size=[0, 9],
    drop_extra_pre_encoded=2,
    last_channel_num=0,
    last_time_num=0
)

cache_last_channel = create_tensor(
    (
        model_config.n_layers, # 24
        batch_size,            # 1
        att_context_size[0],   # 70 self.streaming_cfg.last_channel_cache_size
        model_config.d_model,  # 1024
    )
)
cache_last_time = create_tensor(
    (
        model_config.n_layers,             # 24
        batch_size,                        # 1
        d_model,                           # 1024
        # last_time_cache_size
        model_config.conv_context_size[0], # 8
    ),
    device=device,
    dtype=dtype,
)
cache_last_channel_len = torch.zeros(batch_size) # 1

audio_signal [1, 17, 128] ->
pre_encode [1, 3, 1024] ->
# drop_extra_pre_encoded = streaming_cfg.pre_encode_cache_size // self.subsampling_factor = 2
drop_extra_pre_encoded [1, 1, 1024] ->
pos_enc(cache_len) ([1, 1, 1024], [1, 141, 1024])

