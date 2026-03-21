# EARC Runbook — Current Status and Next Actions

This file captures the minimal context needed to continue work in a new session or location.

## Current State
- Byte-level exporter added: `data/export_fineweb_bytes.py` (validated; val docs = 50,000).
- Minimal EARC training script: `records/track_10min_16mb/2026-03-19_EARC/train_gpt.py`.
  - Single weight-tied block, `RECURRENCE_STEPS` env, SwiGLU(2×), AdamW+cosine, BF16 autocast.
  - Tokenizer-agnostic eval: `BYTE_LEVEL=1` short-circuits byte counting; SP path uses SentencePiece LUTs.
  - INT8+zlib artifact creation + roundtrip eval.
  - Logs: param counts, grad norms (pre-clip), tok/s and B/s at validation.
- Experiment 0A (BPB eval audit): PASS. Eval uses raw-byte denominator for both byte-level and SP-1024 via LUTs; INT8+zlib roundtrip re-eval implemented and logged.

## Experiment 0B — Byte-level vs SP-1024 (Decision)

Runs (ITERATIONS=1000, TRAIN_BATCH_TOKENS=131072, SEQ_LEN=1024, 30 min cap):
- Byte-level (BYTE_LEVEL=1): val_bpb=2.5248; roundtrip_bpb=2.5608; int8+zlib size=846,512 B; tok/s≈289k.
- SP-1024 (BYTE_LEVEL=0): val_bpb=2.5986; roundtrip_bpb=2.6308; int8+zlib size=856,561 B; tok/s≈316k (≈770k B/s).

Decision: proceed with byte-level (wins by ~0.07 BPB; threshold=0.005).

## Experiment 0C — Attention Residuals (AttnRes)

Setup: Byte-level, T=8, identical wallclock (20 min), same hyperparams.

- Control (ATTNRES=0):
  - final_int8_zlib_roundtrip val_bpb=2.1712
  - Serialized model int8+zlib: 709,783 bytes
- AttnRes (ATTNRES=1):
  - final_int8_zlib_roundtrip val_bpb=2.3139
  - Serialized model int8+zlib: 717,172 bytes

Decision: EXCLUDE AttnRes. BPB regresses by +0.1427 and size increases ~7.4 KB.
- .gitignore excludes `data/datasets/`, `logs/`, and `final_model.*` to avoid large file commits.

## Data Prep (1×H100 host)
```bash
# Byte-level shards (10 train shards + full val)
python3 data/export_fineweb_bytes.py --output-dir ./data/datasets/fineweb_bytes --train-shards 10
# SP-1024 shards + tokenizer (10 train shards + full val)
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
```

## Calibration Runs (5 minutes each)
Byte-level:
```bash
RUN_ID=earc_calib_byte \
RECURRENCE_STEPS=8 \
VOCAB_SIZE=256 \
BYTE_LEVEL=1 \
DATA_PATH=./data/datasets/fineweb_bytes \
MAX_WALLCLOCK_SECONDS=300 \
VAL_LOSS_EVERY=200 \
GRAD_CLIP_NORM=1.0 \
python3 records/track_10min_16mb/2026-03-19_EARC/train_gpt.py
```
SP-1024:
```bash
RUN_ID=earc_calib_sp \
RECURRENCE_STEPS=8 \
VOCAB_SIZE=1024 \
BYTE_LEVEL=0 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
MAX_WALLCLOCK_SECONDS=300 \
VAL_LOSS_EVERY=200 \
GRAD_CLIP_NORM=1.0 \
python3 records/track_10min_16mb/2026-03-19_EARC/train_gpt.py
```

## Capture These Metrics
- param_count_total / embed / block (first log lines)
- loss at steps ~1, 50, and 200 (or final)
- `val_bpb` if printed
- `final_model.int8.ptz` size (bytes)
- tokens/sec and bytes/sec (from validation logs)
- grad norm range (pre-clip), any NaNs
- size_delta_pct between configs (if >10%, bump SP `MODEL_DIM` by +32 and re-run 5m SP)

## Next Steps (after calibration)
- If sizes are within 10% and training is stable, launch 90-minute runs with `VAL_LOSS_EVERY=500`.
- If grad_norm clips every step at `lr=1e-3`, lower to `3e-4` for gating runs.
