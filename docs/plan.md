# Parameter Golf Submission Proposal
## Entropy-Aware Recurrent Compression (EARC)
### A Unified Framework for Parameter-Constrained Language Modeling

**Prepared by:** The Consortium  
**Date:** March 19, 2026  
**Target:** OpenAI Parameter Golf Challenge — 16MB Track  
**Submission Tracks:** SOTA Record + Non-Record Architectural Contribution  
**Challenge window:** March 18 – April 30, 2026

---

## Abstract

We propose **EARC** (Entropy-Aware Recurrent Compression), a language model architecture and training framework designed from first principles around the parameter golf constraint. The central insight is that the 16MB compressed artifact limit reframes language modeling as a joint optimization over two compression problems simultaneously: compressing the *data* (minimizing bits-per-byte on FineWeb) and compressing the *model* (minimizing the description length of the weights). Standard training treats these as sequential — train a good model, then compress it. EARC treats them as a single objective.

EARC combines five components, each independently motivated and jointly synergistic:

1. **Byte-level tokenization** — maximize parameter budget allocation to computation rather than vocabulary
2. **Depth-recurrent weight-tied transformer** with phase-modulated step encoding — achieve effective computational depth at zero parameter cost
3. **Attention Residuals (AttnRes)** on the recurrent block's residual stream — content-adaptive depth aggregation across recurrence steps
4. **Entropy-constrained training** via L1 regularization with adaptive dual λ update — align the model's weight distribution with its own compressed representation
5. **Eval-time extended recurrence** — exploit the dedicated 10-minute eval compute window by running more recurrence steps at inference than at training, extracting free BPB improvement at zero parameter cost

We target matching or beating the 4-hour unlimited-compute baseline (1.2074 BPB) within the 10-minute training constraint, and establishing a new SOTA on the 10-minute leaderboard.

---

## 1. Problem Framing

### 1.1 The Constraint Structure

The parameter golf challenge scores submissions on bits-per-byte (BPB) on the FineWeb validation set — tokenizer-agnostic, measured in raw bytes of text — subject to:

```
code_bytes + compressed_model_bytes ≤ 16,000,000
```

where compression uses the challenge's standard INT8 + zlib pipeline (`final_int8_zlib_roundtrip`). Training is capped at 10 minutes on 8×H100 SXM for leaderboard submissions. Evaluation is *separately* capped at 10 minutes on the same hardware — an additional compute budget that the rules explicitly encourage competitors to exploit aggressively.

Most participants will treat this as: *train the best model that fits in 16MB.* This is the wrong framing. The correct framing is:

> **Minimize BPB subject to the compressed representation of the model being ≤ 16MB, and exploit the full eval compute window.**

These are meaningfully different optimization problems. The first ignores the interaction between model quality and model compressibility, and leaves eval compute entirely on the table. The second treats compressibility as a first-class training constraint and eval-time computation as a free resource.

### 1.2 The Rate-Distortion Interpretation

Let D = BPB (distortion) and R = compressed model size (rate). The challenge asks us to minimize D subject to R ≤ 16MB. This is precisely a **rate-distortion optimization problem** — the same framework used in learned image compression.

State-of-the-art learned image codecs do not train a compressor and then apply a separate codec. They train the compressor jointly with a differentiable approximation of the rate term. We apply the same principle to language model weights:

```
L_total = L_BPB(θ) + λ · R(θ)
```

where R(θ) approximates the compressed size of the model weights θ, and λ is a Lagrange multiplier enforcing the 16MB constraint. Section 4 describes our approximation of R(θ) and the adaptive update rule for λ.

### 1.3 The Eval Compute Opportunity

The rules state:

> *"We won't accept submissions that take more than 10 minutes on 8×H100 to evaluate. We encourage competitors to push the bounds of evaluation methods as aggressively as with training methods."*

This is an explicit invitation. 10 minutes on 8×H100 is approximately 2.4×10¹⁸ FLOPS of dedicated eval compute that most competitors will leave unused. Our recurrent architecture naturally exploits this: the number of recurrence steps T is a pure inference-time parameter. The stored model is identical whether we run T=16 or T=128 — only the eval-time compute changes. More recurrence steps mean deeper computation per token, potentially improving BPB at zero parameter cost.

### 1.4 Why the Baseline Leaves Performance on the Table

The baseline (1.2244 BPB) trains a standard 9-layer 512-dim transformer with 1024 BPE vocabulary and applies no compression-aware training. The gap to the 4-hour baseline (1.2074 BPB) — achieved simply by training longer — confirms the 10-minute constraint is binding on data throughput. Our approach attacks this differently: rather than needing more training time, we want each parameter to do more work per compressed byte, and each eval FLOP to contribute to BPB reduction.

---

## 2. Architecture: EARC

### 2.1 Component 1 — Byte-Level Tokenization

**What:** Replace the 1024-token BPE vocabulary with raw bytes (256 tokens).

**Why:**

The embedding table at 1024 tokens × 512 dimensions costs approximately 2MB of the 16MB budget in FP32. At 256 tokens × 512 dimensions this drops to 0.5MB — freeing 1.5MB for recurrent computation. The freed budget scales the main model by roughly 12%, which at our parameter scale translates directly to BPB improvement.

Beyond the budget argument: FineWeb contains HTML, code, multilingual fragments, and structured data — all of which cause systematic tokenization artifacts with BPE vocabularies. Byte-level models handle these natively. The per-step prediction task is also simpler (8 bits of information per step vs ~10 bits for 1024-token BPE), which benefits recurrent models with limited hidden state capacity — each step is easier to get right, and the recurrent hidden state has more steps to refine before each character boundary.

**Rules compliance:** Allowed. The rules explicitly flag tokenizer changes for extra scrutiny on the BPB calculation. We address this directly in Section 5, Experiment 0A — a hard gate that audits the evaluation code before any GPU spend.

**Conditional on Experiment 0B:** We will benchmark byte-level vs BPE-1024 at identical compressed model sizes before committing. If BPE wins by more than 0.005 BPB, we switch to BPE and scale up the model to fill the freed budget.

### 2.2 Component 2 — Depth-Recurrent Weight-Tied Transformer with Phase Modulation

**What:** A single transformer block whose weights are shared (tied) across T recurrent applications. At each recurrence step t, a learned phase signal φ(t) is injected into the residual stream before the block computation.

**Why — weight tying:**

Weight tying across recurrence steps gives effective computational depth of T layers at the parameter cost of 1 layer. For a 512-dim transformer block:

```
MHA (8 heads, 4 KV heads):   4 × 512² = 1,048,576 params
FFN (SwiGLU, 4× multiplier): 8 × 512² = 2,097,152 params  
RMSNorm + misc:              ~5,000 params
Total per block:             ~3,150,000 params (~12.6MB FP32)
```

Under INT8 + entropy-constrained training, expected compressed size: ~2-2.5MB. Running T=16 recurrence steps gives effective depth of 16 layers from a ~2.5MB compressed budget. This is the core parameter efficiency argument.

Critically, weight tying means the *stored* model is the single block. Recurrence depth is free in terms of artifact size.

**Why — phase modulation:**

Naive weight-tied recurrence risks rank collapse — the hidden state converges to an attractor after a few steps rather than continuing to refine. Phase modulation breaks this symmetry:

```
h_t = Block(h_{t-1} + φ(t))

φ(t) = Σ_{k=1}^{K} a_k · sin(ω_k · t + ψ_k)
```

with learnable amplitudes a_k, frequencies ω_k, and phases ψ_k. With K=8 components: 24 scalar parameters — negligible. Different recurrence steps see different phase offsets, allowing early steps to specialize in local patterns and later steps in long-range coherence. This is the HRM two-timescale intuition achieved with a single parameter set and a cheap step-index signal.

**Eval-time recurrence scaling:** Because T is a pure inference parameter, we train with T_train and evaluate with T_eval ≥ T_train. Section 2.5 covers this in detail.

### 2.3 Component 3 — Attention Residuals (AttnRes)

**What:** Replace standard additive residual connections in the recurrent block with AttnRes (Chen et al., Kimi Team, 2026). Each recurrence step's output is a weighted sum over all preceding hidden states, where weights are computed via softmax attention with a learned pseudo-query vector:

```
h_t = Σ_{i=0}^{t-1} α_{i→t} · v_i

α_{i→t} = softmax(w_t · K_{0:t-1})
```

where w_t ∈ ℝ^d is a learnable pseudo-query per step, and K are the normalized prior hidden states.

**Why:** Standard residual connections give equal weight to all prior hidden states via uniform accumulation. As recurrence depth grows, this causes the residual stream magnitude to grow as O(T), progressively diluting early-step contributions — exactly the PreNorm dilution problem identified by the Kimi paper. AttnRes solves this by making aggregation weights input-dependent and content-adaptive.

In our recurrent setting, AttnRes gives the model a form of working memory over its own computational history. For byte-level models, this is particularly valuable: the model can learn to attend back to the hidden state from several bytes ago when encountering a multi-byte character or token boundary, rather than having all prior computation uniformly smeared into the residual stream.

**Memory cost:** For T=16 steps and d=512, storing all prior hidden states costs 16 × 512 × float16 = 16KB per token position. Entirely tractable.

**Parameter cost:** One RMSNorm and one pseudo-query w_t ∈ ℝ^d per recurrence step. For T=16: 16 × (512 + 512) ≈ 16K parameters. Negligible.

**Conditional on Experiment 0C:** We verify AttnRes improves BPB at shallow depth before including it.

### 2.4 Component 4 — Entropy-Constrained Training

**What:** Add a rate term to the training objective:

```
L_total = L_BPB(θ) + λ_t · ||θ||₁
```

**Why L1 specifically:** The L1 norm is the exact entropy penalty under a zero-mean Laplacian prior on weights — the distribution that minimizes description length for weights concentrated near zero. Trained neural network weights are empirically well-approximated by a Laplacian distribution, making this prior natural rather than arbitrary. Under INT8 quantization followed by zlib, a sharper Laplacian weight distribution compresses 20-40% better than a flatter one — this is the compression gain we're buying.

We deliberately choose L1 over a learned entropy model. A learned entropy model (a small neural network modeling the weight distribution, updated jointly with the main model) would give theoretically tighter compression estimates but introduces a second optimization loop that can destabilize training. L1 is the closed-form solution under a Laplacian prior, requires zero additional parameters, and adds negligible compute per step.

**The adaptive dual update rule for λ:**

Rather than hand-tuning λ, we use a feedback controller measuring actual compressed model size at each checkpoint:

```python
# Run after every checkpoint save during λ-active phase:
compressed_size = measure_int8_zlib_size(model)
target_size = 15_500_000  # 500KB headroom for code
size_error = compressed_size - target_size
λ = λ * exp(α · size_error / target_size)  # α = 0.1
```

This is a multiplicative proportional controller on the compression constraint. It converges to exactly 16MB without manual λ sweeps and is robust to architecture changes that affect the base compression ratio.

**The λ annealing schedule:**

λ starts at 0 for the first 80% of training steps. The model learns to predict text without compression pressure. At 80% of steps, λ begins updating via the dual rule. This prevents L1 penalty from interfering with early representation learning while ensuring final weights are compression-optimized.

```
Steps 0   → 0.8×N:  λ = 0          (learn to predict)
Steps 0.8×N → N:   λ updated by dual rule  (learn to compress)
```

**Weight storage optimization:** Before applying INT8 + zlib at artifact creation time, weight tensors are sorted by magnitude within each layer. This increases local similarity of adjacent values in the byte stream, improving LZ77's ability to find repeated patterns. Zero-cost post-processing step.

**Conditional on Experiment 0D:** We verify L1 regularization actually improves zlib compression ratio before committing to this approach.

### 2.5 Component 5 — Eval-Time Extended Recurrence

**What:** Train with T_train recurrence steps. At evaluation time, run T_eval > T_train steps through the same stored model weights.

**Why this is free:** The stored artifact is identical regardless of T_eval. Only the eval-time compute budget changes. With 10 minutes of dedicated eval compute on 8×H100, we have approximately 2.4×10¹⁸ FLOPS available — enough to run substantially more recurrence steps per token than during training.

**The hypothesis:** More recurrence steps at eval time means deeper computation per token, allowing the model to refine its predictions further before committing. The phase modulation (Section 2.2) means each additional step sees a different phase signal and doesn't simply repeat the same computation. The AttnRes mechanism (Section 2.3) allows each step to selectively integrate prior computation.

**The risk:** There is a saturation point beyond which additional recurrence steps stop improving BPB. There may also be instability if T_eval far exceeds T_train (the model may not have learned to use steps beyond T_train effectively). We characterize this empirically in Experiment 0E.

**Implementation:** At eval time, sweep T_eval ∈ {T_train, 2×T_train, 4×T_train, 8×T_train}. Profile wall-clock time per T on 8×H100. Select the largest T_eval that fits within the 10-minute eval budget with reasonable margin.

---

## 3. Full Architecture Specification

```
Model:               EARC-16
Vocabulary:          256 (byte-level, raw UTF-8 bytes)
Hidden dimension:    512
Recurrence steps:    T_train = 16, T_eval = TBD (Experiment 0E)
Attention heads:     8
KV heads:            4 (grouped-query attention)
FFN multiplier:      4× (SwiGLU activation)
Phase encoding:      K=8 sinusoidal components, learnable (a_k, ω_k, ψ_k)
Normalization:       RMSNorm (pre-norm)
Residual:            AttnRes (full, over all T prior hidden states)
Weight tying:        Full — identical weights for all T recurrence steps
Embedding:           Byte-level input, tied with output projection

Parameter count (stored):
  Embedding table:      256 × 512     =    131,072
  Transformer block:                  ~  3,150,000
  Phase encoding:       24 scalars    =         24
  AttnRes queries:      16 × 512      =      8,192
  Output projection:    tied with embedding
  Total:                              ~ 3,290,000 parameters

Size estimates:
  Raw FP32:             ~13.2 MB
  INT8 quantized:       ~3.3 MB
  INT8 + zlib (no L1):  ~2.8 MB (estimated)
  INT8 + zlib (with L1):~2.2 MB (estimated, 20% compression gain)
  Code bytes:           ~100 KB
  Total artifact:       ~2.3 MB

Headroom vs 16MB limit: ~13.7 MB
```

**Note on headroom:** The ~13.7MB of headroom means we can substantially scale up if the byte-level architecture underperforms. If Experiments 0B-0D show our estimates are too optimistic, we have room to double the hidden dimension (512→1024) or increase recurrence steps (T=16→T=32) while staying under budget.

---

## 4. Training Configuration

```
Optimizer:         Muon (proven in NanoGPT speedrun)
Learning rate:     3e-3 with cosine decay to 1e-4
Warmup:            500 steps linear
Gradient clipping: 1.0
Batch tokens:      ~500K tokens/step (calibrate to 8×H100 throughput)
Sequence length:   2048 bytes (byte-level) 
Training data:     FineWeb byte-level variant
                   (export from sp1024 dataset or raw bytes — see data/README.md)
λ schedule:        0 for steps 0→0.8N, dual update for 0.8N→N
Compression check: every 200 steps during λ-active phase
Weight sort:       Magnitude sort applied post-training at artifact creation
Mixed precision:   BF16 training, INT8 for artifact compression
```

**Throughput estimate:** At T=16 recurrence steps, each forward pass has 16× the compute of a single-layer model but uses the same memory (weight tying means no additional parameters loaded). Effective compute per token is similar to a 16-layer standard transformer. On 8×H100 at ~300 TFLOPS effective per GPU, we expect approximately 1-2B byte-level tokens processed in 10 minutes — sufficient for meaningful training given the recurrent depth.

---

## 5. Experiment Zero: Five Prerequisite Experiments

Before the main training run on 8×H100, we execute five cheap experiments on a single H100 to de-risk our five largest assumptions. Each runs for at most 2 hours. Results gate the final architecture decision. **No 8×H100 compute is spent until all five complete.**

### Experiment 0A — BPB Evaluation Code Audit *(Hard Gate)*

**What:** Read `train_gpt.py` evaluation code. Verify the BPB denominator is raw bytes, not tokens. Confirm the `final_int8_zlib_roundtrip` measurement is reproducible and matches the challenge's scoring.

**Why first:** If byte-level tokenization introduces a bug in the BPB calculation (e.g., the eval code accidentally uses token count instead of byte count as the denominator), our byte-level model would appear to achieve artificially better BPB and be disqualified. This is a 1-hour code audit. It costs nothing and gates everything.

**Pass criteria:** BPB calculation verified as raw-byte-denominated and correct for arbitrary vocabularies.

**Fail action:** If the eval code is buggy for non-standard vocabularies, we implement a corrected BPB calculation and document it thoroughly in our submission README, per the rules' requirement to "prove with certainty that the val_bpb is correctly calculated."

### Experiment 0B — Byte-Level vs BPE-1024

**Setup:** Train identical weight-tied recurrent models (T=8, d=256, 2 hours on 1×H100) at byte-level vs BPE-1024. Scale the byte-level model up slightly if needed to ensure identical compressed artifact sizes.

**Measure:** BPB on FineWeb validation, compressed model size.

**Decision rule:**
- Byte-level BPB ≤ BPE BPB → proceed with byte-level
- BPE BPB < byte-level BPB by >0.005 → switch to BPE-1024
- Difference <0.005 → proceed with byte-level (parameter budget reasons)

**Expected outcome:** Byte-level wins due to embedding table savings enabling larger model and simpler per-step prediction.

### Experiment 0C — AttnRes at Shallow Depth

**Setup:** Train T=8, d=256 model with and without AttnRes (2 hours on 1×H100).

**Measure:** BPB delta, training stability (gradient norms, loss curves).

**Decision rule:**
- AttnRes improves BPB by >0.001 → include in main architecture
- No improvement → drop AttnRes; recover parameter overhead for main model

**Expected outcome:** AttnRes helps. PreNorm dilution is proportionally worse at shallow depth. AttnRes across recurrence steps (depth axis) is a natural fit for recurrent models.

### Experiment 0D — L1 Compression Delta

**Setup:** Train T=8, d=256 model with λ ∈ {0, 1e-5, 3e-5, 1e-4} (four short runs, 30 min each on 1×H100).

**Measure:** For each λ: (a) BPB on validation, (b) compressed model size (INT8 + zlib).

**Decision rule:**
- L1 at best λ reduces compressed size by >15% with BPB degradation <0.003 → include entropy training
- Compression gain <15% → reconsider entropy training
- BPB degradation >0.003 → lower λ target or remove entropy training

**Expected outcome:** L1 at λ≈3e-5 reduces compressed size by 20-35% with minimal BPB impact. This is the critical validation of the central thesis.

**Bonus:** The λ sweep gives us a rough calibration point for the dual update rule's starting value.

### Experiment 0E — Eval-Time Recurrence Scaling

**Setup:** Train T=8, d=256 model normally. Evaluate at T_eval ∈ {8, 16, 32, 64, 128}. Measure BPB and wall-clock time per T_eval setting.

**Measure:** BPB vs T_eval curve. Saturation point. Time per T_eval on 1×H100 (scale to 8×H100 estimate).

**Decision rule:**
- If BPB continues improving at T_eval = 2×T_train: set T_eval = 4×T_train for main run
- If BPB saturates at T_eval = T_train: eval-time scaling doesn't help, remove from proposal
- Use time measurements to select largest T_eval fitting in 10-minute eval budget

**Expected outcome:** BPB improves up to T_eval ≈ 3-4×T_train, then saturates. Phase modulation should delay saturation vs naive weight tying.

---

## 6. Ablation Sequence

The main paper ablation follows this additive sequence, each step evaluated at identical compressed artifact size. This structure is designed to be the paper:

| Step | Configuration | Expected BPB |
|------|--------------|--------------|
| 0 | Baseline (9L 512D 1024-BPE standard transformer) | 1.2244 |
| 1 | + Weight tying (T=16 recurrence, same stored params) | ~1.210 |
| 2 | + Phase modulation | ~1.205 |
| 3 | + AttnRes across recurrence steps | ~1.201 |
| 4 | + Byte-level tokenization | ~1.193 |
| 5 | + Entropy training (L1 + dual λ) | ~1.185 |
| 6 | + Eval-time extended recurrence (T_eval > T_train) | ~1.178 |
| **Target** | **Full EARC** | **≤ 1.185 (training) / ≤ 1.178 (with eval scaling)** |

The target of ≤1.185 BPB from training alone beats the 4-hour unlimited-compute baseline (1.2074) by ~0.022 BPB — achieved within the 10-minute training constraint. The eval-time recurrence adds further improvement using the dedicated eval compute window.

---

## 7. Statistical Validation Protocol

Per challenge rules, any SOTA record must demonstrate p<0.01 that the improvement exceeds 0.005 nats over the current SOTA.

**Protocol:**
- Minimum 8 full training runs from independent random seeds
- Report mean, standard deviation, min, max, and 99% confidence interval on BPB
- If run-to-run standard deviation σ > 0.003 BPB, increase to 16 runs
- Statistical test: one-sided Welch t-test against current SOTA (1.2244)
- Log the λ trajectory for each run to verify convergence to target compressed size
- Report both training-only BPB and eval-with-extended-recurrence BPB separately

**Variance sources to monitor:**
- λ dual update rule trajectory (depends on training dynamics)
- Phase encoding initialization (random a_k, ω_k, ψ_k at init)
- Muon optimizer stochasticity

---

## 8. Risk Register

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Byte-level BPB eval code is buggy | Low | High — disqualification | Experiment 0A hard gate; document BPB calculation explicitly |
| Byte-level loses to BPE at this scale | Medium | Medium — fallback to BPE | Experiment 0B gates this; BPE fallback scales model to fill budget |
| Rank collapse in recurrence despite phase modulation | Low-Medium | High | Monitor hidden state rank during training; add noise injection if needed; reduce T |
| L1 hurts BPB more than it helps compression | Low | Medium | Experiment 0D gates this; fallback to no entropy training |
| AttnRes overhead slows training throughput materially | Low | Low | Profile on day one; Block AttnRes available as fallback |
| Eval-time recurrence saturates at T_train | Medium | Low | Experiment 0E gates this; default to standard eval if no gain |
| λ dual update destabilizes late training | Low | Medium | Cap λ at λ_max = 1e-4; monitor gradient norms; emergency fallback to fixed λ |
| Run-to-run variance too high for statistical significance | Medium | Medium | Budget 16 runs if σ > 0.003; 8×H100 time is the bottleneck |
| Headroom estimate wrong (model doesn't compress to ~2.5MB) | Low | Medium | Scale down hidden dim; reduce T; fallback to BPE to recover 1.5MB |

---

## 9. What We Are Not Doing, And Why

**Hypernetwork weight expansion:** Compelling theoretically but requires indirect training signal through the hypernetwork. At 10 minutes of training compute convergence is not guaranteed. Deferred to future work — potentially valuable as a non-record submission on the unlimited compute track.

**Learned entropy model:** 90% of the compression benefit of a learned entropy model is captured by L1 under a Laplacian prior. The remaining 10% is not worth the additional optimization complexity and potential instability.

**Test-time gradient adaptation:** Technically allowed by the rules (no restriction on eval-time computation, only on training data access). We achieve the same goal more cleanly via eval-time extended recurrence, which requires no gradient computation and has no instability risk.

**Document routing / domain classifier:** Potentially valuable for heterogeneous eval sets. Adds ~10K parameter overhead and requires pseudo-label training. Deferred — add as a non-record supplementary submission if main architecture underperforms on specific FineWeb sub-domains.

**Custom ANS codec:** Would improve compression beyond zlib, effectively expanding the 16MB budget. Adds ~2-5KB of code bytes (acceptable) but engineering complexity is high. Deferred to v2 — implement if v1 architecture underperforms and we need to recover headroom.

**MoE, Mamba, BitNet:** All reasonable ideas that will be well-represented by other teams. We deliberately avoid the crowded part of the search space.

---

## 10. Submission Artifacts

Per challenge rules, the submission PR must contain exactly:

```
records/track_10min_16mb/2026-XX-XX_EARC/
  README.md           — this document, condensed
  submission.json     — name, GitHub ID, val_bpb, metadata
  train_log.txt       — auto-produced by train_gpt.py
  train_gpt.py        — complete self-contained training script
```

The `train_gpt.py` script must:
- Contain all code including weight sorting, L1 regularization, dual λ update, AttnRes, phase encoding
- Produce a reproducible artifact when run with a fixed seed
- Complete training in under 10 minutes on 8×H100 SXM
- Complete evaluation in under 10 minutes on 8×H100 SXM
- Require no external downloads, network calls, or training data access during evaluation
- Report `val_bpb` computed as total nats / total raw bytes (byte-denominated, tokenizer-agnostic)

---

## 11. Timeline

| Day | Activity | Owner |
|-----|----------|-------|
| 1 | Experiment 0A: Audit eval code, verify BPB calculation | Jang + Agarwal |
| 1 | Implement base recurrent model, training infrastructure | Karpathy |
| 2 | Experiments 0B, 0C, 0D, 0E in parallel on 1×H100 | All |
| 3 | Analyze experiment results, finalize architecture | Consortium |
| 3 | Scale decisions: hidden dim, T_train, T_eval target | Dean + Kaplan |
| 4-5 | Implement full EARC in train_gpt.py | Karpathy + Yang Zhilin + Pachocki |
| 6 | First full training run on 8×H100, check BPB and artifact size | All |
| 7 | Diagnose and fix any issues from run 1 | Karpathy + Pachocki |
| 8-10 | Ablation sequence (Steps 1-6), one per day | Yang Zhilin + Kaplan |
| 11-13 | Statistical validation: 8-16 seed runs | Hobbhahn + Agarwal |
| 13 | Per-domain BPB breakdown on FineWeb sub-types | Jang + Barzilay |
| 14 | Prepare and submit PR | Woodward + Korinek |

**Total: 14 days to submission. Challenge closes April 30 — 6 weeks of buffer.**

---

## 12. Author Responsibilities

| Person | Primary Role |
|--------|-------------|
| Karpathy | Implementation lead, train_gpt.py, training infrastructure |
| Yang Zhilin | AttnRes integration, recurrent architecture design |
| Pachocki | Optimizer configuration, training stability, λ dynamics |
| Donti | Dual λ update rule, rate-distortion theoretical framing |
| Dean | Throughput optimization, scaling estimates, compute budgeting |
| Agarwal | Compression benchmarking, artifact pipeline, INT8+zlib audit |
| Sutskever | Byte-level architecture, theoretical framing, eval recurrence |
| Kaplan | Scaling law analysis, ablation design, architecture sizing |
| Hobbhahn | Statistical validation protocol, significance testing |
| Barzilay | Evaluation analysis, per-domain BPB breakdown |
| Jang | Eval harness, FineWeb characterization, BPB code audit |
| Ha | Compressed sensing baseline (non-record unlimited compute track) |
| Korinek | Paper framing, submission narrative, README |
| Kokotajlo | Representational analysis of entropy-trained weights |
| Woodward | Project management, scope enforcement, PR submission |

---

## 13. Theoretical Contribution

Independent of the leaderboard result, EARC makes a conceptual contribution worth writing up regardless of placement:

**The central claim:** When the evaluation metric is bits-per-byte and the constraint is compressed model size, the compression of a language model and the compression of language are the *same* optimization problem. A model should be trained to minimize its own description length jointly with the description length of the data it models.

This reframing — from "train a model, then compress it" to "train a model that is its own compressed representation" — is established in image compression but has not been systematically applied to language model training in a setting where artifact size is literally the optimization constraint.

The ablation structure isolates the contribution of each component to closing the gap between 10-minute and unlimited-compute training. The eval-time recurrence analysis characterizes the relationship between inference compute and language modeling quality in the weight-tied recurrent regime — a relationship that is underexplored in the literature.

**Secondary finding to watch for:** Does entropy-constrained training produce qualitatively different internal representations? Kokotajlo will analyze weight distributions, attention patterns, and hidden state geometry in entropy-trained vs standard-trained models. If the answer is yes — if compressibility pressure produces more structured, interpretable representations — that is a finding with implications beyond this competition.

---

*"We treat the compression of the model and the compression of the data as the same optimization problem. The 16MB limit is not a constraint on our model. It is the model."*
