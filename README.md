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
- [x] **9.** shift_targets_right
- [x] **10.** __init__
- [x] **11.** __init__
- [x] **12.** __init__
- [x] **13.** __init__
- [ ] **14.** encoder_layer_forward
- [ ] **15.** __init__
- [ ] **16.** decoder_layer_forward
- [ ] **17.** __init__
- [ ] **18.** __init__
- [ ] **19.** __init__
- [ ] **20.** tie_target_embedding
- [ ] **21.** __init__
- [ ] **22.** label_smoothing_distribution
- [ ] **23.** loss_ignoring_pad
- [ ] **24.** __init__
- [ ] **25.** noam_learning_rate
- [ ] **26.** make_optimizer
- [ ] **27.** optimizer_hyperparameters
- [ ] **28.** transformer_training_loss
- [ ] **29.** backward_step
- [ ] **30.** train_batch
- [ ] **31.** evaluate_batch
- [ ] **32.** checkpoint_roundtrip
- [ ] **33.** greedy_next_token
- [ ] **34.** greedy_decode
- [ ] **35.** greedy_decode_eos
- [ ] **36.** beam_expand_scores
- [ ] **37.** beam_topk
- [ ] **38.** update_finished_beams
- [ ] **39.** length_penalty
- [ ] **40.** beam_decode_step
- [ ] **41.** beam_decode
- [ ] **42.** tiny_model_inference
- [ ] **43.** end_to_end_decode
