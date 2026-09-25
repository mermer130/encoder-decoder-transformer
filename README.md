# 从0到1实现Transformer架构

手把手教你实现Transformer的完整架构

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** bpe_tokenize
- [x] **2.** build_token_id_matrix
- [x] **3.** __init__
- [x] **4.** sinusoidal_encoding
- [x] **5.** __init__
- [x] **6.** make_src_mask
- [x] **7.** subsequent_mask
- [ ] **8.** make_tgt_mask
- [ ] **9.** shift_targets_right
- [x] **10.** __init__
- [x] **11.** mha_attn_kernel
- [ ] **12.** __init__
- [ ] **13.** residual_dropout
- [ ] **14.** __init__
- [x] **15.** __init__
- [ ] **16.** encoder_layer_forward
- [ ] **17.** __init__
- [ ] **18.** decoder_layer_forward
- [ ] **19.** __init__
- [ ] **20.** __init__
- [ ] **21.** __init__
- [ ] **22.** tie_target_embedding
- [ ] **23.** __init__
- [ ] **24.** label_smoothing_distribution
- [ ] **25.** loss_ignoring_pad
- [ ] **26.** __init__
- [ ] **27.** noam_learning_rate
- [ ] **28.** make_optimizer
- [ ] **29.** optimizer_hyperparameters
- [ ] **30.** transformer_training_loss
- [ ] **31.** backward_step
- [ ] **32.** train_batch
- [ ] **33.** evaluate_batch
- [ ] **34.** checkpoint_roundtrip
- [ ] **35.** greedy_next_token
- [ ] **36.** greedy_decode
- [ ] **37.** greedy_decode_eos
- [ ] **38.** beam_expand_scores
- [ ] **39.** beam_topk
- [ ] **40.** update_finished_beams
- [ ] **41.** length_penalty
- [ ] **42.** beam_decode_step
- [ ] **43.** beam_decode
- [ ] **44.** tiny_model_inference
- [ ] **45.** end_to_end_decode
