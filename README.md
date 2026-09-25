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
- [x] **5.** sinusoidal_encoding
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
- [ ] **20.** label_smoothing_distribution
- [ ] **21.** loss_ignoring_pad
- [ ] **22.** __init__
- [ ] **23.** noam_learning_rate
- [ ] **24.** make_optimizer
- [ ] **25.** optimizer_hyperparameters
- [ ] **26.** transformer_training_loss
- [ ] **27.** backward_step
- [ ] **28.** train_batch
- [ ] **29.** evaluate_batch
- [ ] **30.** checkpoint_roundtrip
- [ ] **31.** greedy_next_token
- [ ] **32.** greedy_decode
- [ ] **33.** greedy_decode_eos
- [ ] **34.** beam_expand_scores
- [ ] **35.** beam_topk
- [ ] **36.** update_finished_beams
- [ ] **37.** length_penalty
- [ ] **38.** beam_decode_step
- [ ] **39.** beam_decode
- [ ] **40.** tiny_model_inference
- [ ] **41.** end_to_end_decode
