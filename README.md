# Encoder-Decoder Transformer

按 Encoder-Decoder 结构手写完整模型：词嵌入、正弦位置编码、多头注意力、Post-Norm、真实 Adam 与解码。每一步对应一个 nn.Module 或训练/推理接口。

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** __init__
- [ ] **2.** 正弦位置编码矩阵
- [ ] **3.** __init__
- [ ] **4.** 源端 Padding 掩码
- [ ] **5.** 因果下三角掩码
- [ ] **6.** 目标端 Padding 与因果掩码
- [ ] **7.** 目标序列右移
- [ ] **8.** 缩放注意力分数
- [ ] **9.** 注意力掩码填负无穷
- [ ] **10.** 注意力 Softmax
- [ ] **11.** 注意力权重 Dropout
- [ ] **12.** 权重乘 Value
- [ ] **13.** 缩放点积注意力
- [ ] **14.** 最后一维拆成多头
- [ ] **15.** 多头合并回模型维
- [ ] **16.** __init__
- [ ] **17.** self_attention_forward
- [ ] **18.** cross_attention_forward
- [ ] **19.** __init__
- [ ] **20.** 残差分支 Dropout
- [ ] **21.** __init__
- [ ] **22.** FFN 第一层 ReLU
- [ ] **23.** __init__
- [ ] **24.** feed_forward_hidden
- [ ] **25.** feed_forward_forward
- [ ] **26.** 构造编码器注意力
- [ ] **27.** 构造编码器前馈
- [ ] **28.** __init__
- [ ] **29.** encoder_layer_forward
- [ ] **30.** clone_encoder_layer
- [ ] **31.** __init__
- [ ] **32.** encoder_forward
- [ ] **33.** encoder_state_dict
- [ ] **34.** 构造 N 层编码器
- [ ] **35.** 解码器自注意力掩码
- [ ] **36.** decoder_self_attention
- [ ] **37.** decoder_cross_attention
- [ ] **38.** decoder_feed_forward
- [ ] **39.** __init__
- [ ] **40.** decoder_layer_forward
- [ ] **41.** clone_decoder_layer
- [ ] **42.** __init__
- [ ] **43.** decoder_forward
- [ ] **44.** decoder_state_dict
- [ ] **45.** 构造 N 层解码器
- [ ] **46.** decoder_output_shape
- [ ] **47.** __init__
- [ ] **48.** generator_forward
- [ ] **49.** __init__
- [ ] **50.** encode_source
- [ ] **51.** decode_target
- [ ] **52.** encoder_decoder_forward
- [ ] **53.** 源端与目标端嵌入
- [ ] **54.** tie_target_embedding
- [ ] **55.** xavier_initialize
- [ ] **56.** 构造完整 Transformer
- [ ] **57.** __init__
- [ ] **58.** 标签平滑分布
- [ ] **59.** loss_ignoring_pad
- [ ] **60.** __init__
- [ ] **61.** Noam 学习率公式
- [ ] **62.** make_optimizer
- [ ] **63.** optimizer_hyperparameters
- [ ] **64.** transformer_training_loss
- [ ] **65.** backward_step
- [ ] **66.** train_batch
- [ ] **67.** evaluate_batch
- [ ] **68.** checkpoint_roundtrip
- [ ] **69.** greedy_next_token
- [ ] **70.** greedy_decode
- [ ] **71.** greedy_decode_eos
- [ ] **72.** 展开 Beam 分数
- [ ] **73.** 选取 Beam Top-k
- [ ] **74.** 更新 Beam 完成标记
- [ ] **75.** 长度惩罚
- [ ] **76.** beam_decode_step
- [ ] **77.** beam_decode
- [ ] **78.** tiny_model_inference
- [ ] **79.** end_to_end_decode
