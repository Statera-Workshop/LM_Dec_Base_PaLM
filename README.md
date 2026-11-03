# LM_Dec_Base_PaLM

PaLM - Base family, decoder-only byte-level language model, standardised in the Statera Working Group Language Model library.

## Status

- Verification level: **Benchmarked**
- Failure class: none

## Provenance

- Source repository: LM_2022_PaLM
- Upstream / license: vendored from RichardMinsooGo/LM_2022_PaLM (see LICENSE in vendor folder if present)

## Differences from the source ([Unified] protocol)

- parallel blocks, rotary, SwiGLU (upstream defaults)

All models share one protocol: 256-byte vocabulary, dim 512, 6 layers, 8 heads x 64, Adam 2e-4 with 200 warm-up steps, gradient clip 0.5, bf16 autocast (automatic fp32 fallback). Numbers rank structures under one budget on one GPU (RTX 3070 Ti 8 GB); they do not reproduce paper results.

## Results

- Parameters: 22.55 M
- Causality check: pass (largest change of earlier outputs 0.0e+00)
- Overfit check (4 x 256 bytes): 5.547 -> 0.001 nats, pass
- LM-02 enwik8 (200M tokens): valid 1.457 / test 1.442 bits per byte
- Peak GPU memory at length 2048 (batch 1): train 1808 MB, inference 418 MB
- Peak GPU memory at length 8192 (batch 1): train None, inference None
- LM-05 recall `lm05v3`: 4 pairs @ 512: 0.23 (keys removed 0.26); 4 pairs @ 2048: 0.25 (keys removed 0.26); 16 pairs @ 512: 0.07 (keys removed 0.07); 16 pairs @ 2048: 0.07 (keys removed 0.07)
- LM-06 memory update `lm06`: @512: latest 0.71, unchanged 0.04, stale 0.01; @2048: latest 0.07, unchanged 0.04, stale 0.05

## Tests

- check: data counts, causality (no look-ahead, outputs use the current byte), parameters
- scaling: peak GPU memory and time vs length 512-8192
- overfit: memorise 4 x 256 bytes below 0.5 nats
- LM-02: enwik8 next-byte prediction (bits per byte)
- LM-05: long-horizon key-value recall with key-removed control (v3: distinct queries)
- LM-06: memory update - latest value vs stale value after a key is reassigned

## Run

```bash
pip install torch==2.5.1 einops numpy
python run.py check palm
python run.py lm02 palm      # enwik8 is downloaded to data/ on first use
```
