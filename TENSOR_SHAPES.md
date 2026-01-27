# Tensor Shapes for NeMo ASR Model (nemotron-speech-streaming-en-0.6b)

Total tensors: 653

## Model Configuration
```
n_mels       = 128    # mel spectrogram features
d_model      = 1024   # model dimension
n_heads      = 8      # attention heads
d_head       = 128    # head dimension (d_model / n_heads)
d_ff         = 4096   # feedforward dimension
n_layers     = 24     # number of conformer layers
kernel_size  = 9      # conv kernel size (inferred from weights)
vocab_size   = 1025   # vocabulary (1024 tokens + blank)
decoder_dim  = 640    # decoder LSTM hidden size
joint_dim    = 640    # joint network hidden size
```

## Encoder

### ConvSubsampling (encoder.pre_encode)
| Tensor | Shape | Description |
|--------|-------|-------------|
| conv.0.weight | [256, 1, 3, 3] | CausalConv2D(1, 256, k=3, s=2) |
| conv.0.bias | [256] | |
| conv.2.weight | [256, 1, 3, 3] | Depthwise CausalConv2D |
| conv.2.bias | [256] | |
| conv.3.weight | [256, 256, 1, 1] | Pointwise Conv2D |
| conv.3.bias | [256] | |
| conv.5.weight | [256, 1, 3, 3] | Depthwise CausalConv2D |
| conv.5.bias | [256] | |
| conv.6.weight | [256, 256, 1, 1] | Pointwise Conv2D |
| conv.6.bias | [256] | |
| out.weight | [1024, 4352] | Linear(17*256=4352, 1024) |
| out.bias | [1024] | |

### Conformer Layers (encoder.layers.{0-23})
Each layer has the following tensors:

| Tensor | Shape | Description |
|--------|-------|-------------|
| norm_feed_forward1.weight | [1024] | LayerNorm |
| norm_feed_forward1.bias | [1024] | |
| feed_forward1.linear1.weight | [4096, 1024] | FFN1 up-projection |
| feed_forward1.linear2.weight | [1024, 4096] | FFN1 down-projection |
| norm_self_att.weight | [1024] | LayerNorm |
| norm_self_att.bias | [1024] | |
| self_attn.linear_q.weight | [1024, 1024] | Query projection |
| self_attn.linear_k.weight | [1024, 1024] | Key projection |
| self_attn.linear_v.weight | [1024, 1024] | Value projection |
| self_attn.linear_pos.weight | [1024, 1024] | Position projection |
| self_attn.linear_out.weight | [1024, 1024] | Output projection |
| self_attn.pos_bias_u | [8, 128] | Position bias u |
| self_attn.pos_bias_v | [8, 128] | Position bias v |
| norm_conv.weight | [1024] | LayerNorm |
| norm_conv.bias | [1024] | |
| conv.pointwise_conv1.weight | [2048, 1024, 1] | Pointwise + GLU |
| conv.depthwise_conv.weight | [1, 1024, 9] | Depthwise causal conv |
| conv.batch_norm.weight | [1024] | BatchNorm (used as LayerNorm) |
| conv.batch_norm.bias | [1024] | |
| conv.pointwise_conv2.weight | [1024, 1024, 1] | Pointwise |
| norm_feed_forward2.weight | [1024] | LayerNorm |
| norm_feed_forward2.bias | [1024] | |
| feed_forward2.linear1.weight | [4096, 1024] | FFN2 up-projection |
| feed_forward2.linear2.weight | [1024, 4096] | FFN2 down-projection |
| norm_out.weight | [1024] | Final LayerNorm |
| norm_out.bias | [1024] | |

## Decoder (decoder.prediction)

| Tensor | Shape | Description |
|--------|-------|-------------|
| embed.weight | [1025, 640] | Token embedding (vocab_size, embed_dim) |
| dec_rnn.lstm.weight_ih_l0 | [2560, 640] | LSTM layer 0 input-hidden weights [4*hidden, input] |
| dec_rnn.lstm.weight_hh_l0 | [2560, 640] | LSTM layer 0 hidden-hidden weights [4*hidden, hidden] |
| dec_rnn.lstm.bias_ih_l0 | [2560] | LSTM layer 0 input-hidden bias |
| dec_rnn.lstm.bias_hh_l0 | [2560] | LSTM layer 0 hidden-hidden bias |
| dec_rnn.lstm.weight_ih_l1 | [2560, 640] | LSTM layer 1 input-hidden weights |
| dec_rnn.lstm.weight_hh_l1 | [2560, 640] | LSTM layer 1 hidden-hidden weights |
| dec_rnn.lstm.bias_ih_l1 | [2560] | LSTM layer 1 input-hidden bias |
| dec_rnn.lstm.bias_hh_l1 | [2560] | LSTM layer 1 hidden-hidden bias |

**Note:** Decoder output is the LSTM hidden state (640). No separate fc projection layer.

## Joint Network (joint)

| Tensor | Shape | Description |
|--------|-------|-------------|
| enc.weight | [640, 1024] | Encoder projection (joint_dim, encoder_dim) |
| enc.bias | [640] | |
| pred.weight | [640, 640] | Decoder projection (joint_dim, decoder_dim) |
| pred.bias | [640] | |
| joint_net.2.weight | [1025, 640] | Output projection (vocab_size, joint_dim) |
| joint_net.2.bias | [1025] | |

## Positional Encoding

Precomputed sinusoidal embeddings:
| Tensor | Shape | Description |
|--------|-------|-------------|
| pos_emb | [1024, 1023] | [d_model, 2*max_len-1] for max_len=512 |

Stored in NeMo's descending order: positions go from +(max_len-1) down to -(max_len-1).

## GGML Layout Convention

GGML uses column-major storage. When loading weights:
- A PyTorch tensor [A, B, C] maps to GGML ne = [C, B, A]
- For matrix multiply: `ggml_mul_mat(W, x)` computes x @ W.T
- Weight matrices are stored transposed relative to PyTorch: [out_features, in_features]

## LSTM Gates Order

LSTM gates are concatenated in order: [i, f, g, o]
- i: input gate
- f: forget gate
- g: cell gate (tanh)
- o: output gate

Gate computation:
```
gates = x @ W_ih.T + h @ W_hh.T + b_ih + b_hh
i = sigmoid(gates[0:h])
f = sigmoid(gates[h:2h])
g = tanh(gates[2h:3h])
o = sigmoid(gates[3h:4h])
c_new = f * c + i * g
h_new = o * tanh(c_new)
```
