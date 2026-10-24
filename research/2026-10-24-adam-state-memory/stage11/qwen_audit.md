# Independent Qwen parameter and factor-history budget audit

## Results and limits

The architecture-derived count is **26,895,998,464 text parameters**, including **2,645,504 non-dense parameters**. A specifically defined, optimistic shared-input BF16 factor record costs **12,650,504 bytes per accumulated token**, including one 64-bit embedding token ID. At **512 token slots per optimizer step**, the factor payload alone is **6,477,058,048 bytes = 6.0322303772 GiB per step**, or **51,816,464,384 bytes = 48.2578430176 GiB for eight steps**.

These are analytical byte counts, not a Qwen implementation, an exact-gradient proof, an HBM measurement, or a universal lower bound on every conceivable replay algorithm. In particular, 512 is the full-length, single-device scenario implied by batch 1 × four accumulated microbatches × maximum length 128. It is **not a verified count of actual tokens in the earlier run**.

## Source provenance and checkpoint identity

1. The local repository snapshot is pinned to `f98725dedc5a7ea371f265f8a084f3a6c20769a0` in `references/qwen27b-ascend-training.snapshot-manifest.json:2–10`. This audit recomputes all 11 files' Git blob SHA-1 values and sizes against that manifest. All match; checks are in `qwen_audit.json`.
2. The repository's `research/2026-10-23-fp32-adam-memory.md:13–22` reports the text count, frozen vision, approximately 50.96 GiB of loaded text-plus-vision BF16 weights, and exclusion of checkpoint MTP from the native HF loading path. It does not preserve the original tensor headers or model config. Therefore that local statement alone cannot independently establish the parameter count.
3. To close the architecture gap, the official [Qwen3.5-27B config](https://huggingface.co/Qwen/Qwen3.5-27B/raw/af65380a20d418eeb0a2bcb784dd43e9b76c4a2e/config.json) and official [Transformers v5.14.1 source](https://raw.githubusercontent.com/huggingface/transformers/v5.14.1/src/transformers/models/qwen3_5/modeling_qwen3_5.py) were read without modifying any repository or installing/running that source. Exact downloaded bytes are archived in `qwen_audit_sources/`, with URLs, hashes, retrieval timestamps and selected provenance headers in `manifest.json`. The source is Apache-2.0; its copyright header and the upstream `LICENSE_transformers` are preserved.
4. The local evidence names the model **“Qwen3.8-27B (Qwen3_5ForConditionalGeneration)”** and Transformers **5.14.1** (`references/qwen27b-ascend-training/evidence/2026-10-23/qwen27b-swift-sft.json:212–218`). The public Qwen3.5 architecture reproduces its documented count exactly. This is strong structural corroboration, **not authentication that the separately named user's checkpoint has identical tensors, values, local patches, or loading behavior**. No checkpoint or device was accessed in this audit.

Paths below are relative to this `stage11` directory unless prefixed with `references/`.

## Reproducing the parameter count

The archived config (`qwen_audit_sources/qwen3_5_27b_config.json:8–18,19–100,114`) gives:

- Hidden width H=5,120, MLP width I=17,408, vocabulary V=248,320.
- 64 blocks: 48 gated-delta and 16 full-attention blocks.
- Full attention: 24 query heads × 256 = Q=6,144; four KV heads × 256 = K=1,024. The query projection has **2Q** outputs because it also supplies an output gate.
- Gated delta: key width Dk=16×128=2,048; value width Dv=48×128=6,144; Nv=48 value heads; depthwise convolution width four.
- Untied embedding/output weights; no ordinary attention projection biases.

Source line references below refer to the exact archived `modeling_qwen3_5_v5_14_1.py`, not web-rendered line numbering.

| Component | Formula | Parameters | Source lines |
|---|---:|---:|---|
| 64 MLPs | 64×3HI | 17,112,760,320 | 724–736 |
| 48 gated-delta dense projections | 48×H(2Dk+3Dv+2Nv) | 5,560,074,240 | 377–383,419,435–438 |
| 16 full-attention dense projections | 16×H(3Q+2K) | 1,677,721,600 | 649–674 |
| Embedding and untied output head | 2VH | 2,542,796,800 | 1147,1706; config:114 |
| Gated-delta depthwise convolutions | 48×(2Dk+Dv)×4 | 1,966,080 | 391–399 |
| Gated-delta dt_bias and A_log | 48×2Nv | 4,608 | 402–407 |
| Gated-delta normalization weights | 48×128 | 6,144 | 409–416 |
| Full-attention q/k normalization | 16×2×256 | 8,192 | 673–674 |
| Two block normalization weights | 64×2H | 655,360 | 769–771 |
| Final normalization weight | H | 5,120 | 1151 |
| **Total** | | **26,895,998,464** | |

The enumerator in `qwen_audit.py` creates 851 tensor-shape records, saved in `qwen_audit.parameters.csv`, and sums their products independently of the proposed total. It includes embedding, untied lm_head, every dense projection, all listed norms, depthwise convolution kernels, dt_bias and A_log. It excludes visual encoder/aligner parameters, MTP, rotary buffers, caches, and LoRA adapters. MTP keys are explicitly ignored by the native source at line 824. Frozen vision still consumes memory even though it has no trainable moments in this hypothetical text-only full-parameter optimizer.

Two FP32 moments cost 8P = **215,167,987,712 bytes = 200.3908042908 GiB**. The two-byte text-weight idealization is 2P = **50.0977010727 GiB**; that is a separate item, not a replacement-state saving.

## Deriving the optimistic factor payload

For a linear weight, the proposed record stores its input activation X and output derivative dY, so each token contributes input-width plus output-width scalar values. It shares X only across projections which consume the **same tensor**. Distinct inputs, including down-projection inputs and output-projection inputs, are not merged just because their dimensions match.

| Group | Shared-factor scalar elements per token | Across all applicable blocks |
|---|---:|---:|
| MLP: shared gate/up X, two dY vectors, down X and dY | 2H+3I = 62,464 | 3,997,696 |
| Full attention: shared q/k/v X, dY for 2Q/K/K, o X and dY | 2H+3Q+2K = 30,720 | 491,520 |
| Gated delta: shared qkv/z/b/a X, their dY, output X and dY | 2H+2Dk+3Dv+2Nv = 32,864 | 1,577,472 |
| Untied lm_head X and dY | H+V | 253,440 |
| Embedding output derivative | H | 5,120 |
| **Total scalar elements E** | | **6,325,248** |

Embedding uses a token index plus its H-wide derivative rather than storing a V-wide one-hot vector. With every retained scalar assumed to be natively BF16 and one int64 token ID:

F = 2E + 8 = **12,650,504 bytes/token**.

The 8-byte ID is not doubled in the FP32-factor sensitivity case. Giving every projection its own input instead of sharing adds H×(64+2×16+3×48)=1,228,800 scalars/token, or 2,457,600 bytes/token in BF16.

### What this expression omits

- The **2,645,504 non-dense trainables** are included in P but not reconstructed by these linear/embedding factors. One transparent accounting option is to retain their already-aggregated FP32 gradients: **10,582,016 additional bytes per optimizer step**. An alternative is to update their ordinary dense FP32 m/v persistently: **21,164,032 bytes**, plus current gradients/temporaries. The alternative scenario explicitly charges a full FP32 small-gradient buffer of **10,582,016 bytes**, for a fixed **31,746,048-byte** subtotal instead of small-gradient history. Both choices include dt_bias; there are no missing ordinary dense biases because those projections are bias-free. Neither representation is free, and neither is claimed optimal.
- Replay step/microbatch descriptors, shapes, offsets, tensor/version identities, masks or token-length boundaries, loss scaling, clipping factors and optimizer hyperparameters, allocator alignment, and moment/gradient tiles are absent from F. Some can be shared or derived, but their exact requirements require an implementation.
- Forward/backward activation and recurrent-state peaks, attention workspaces, loss/logit buffers, weights, optional FP32 master weights, communication buckets, duplicated/sharded factors, runtime baseline memory and allocator fragmentation are not included. The archived model source itself performs several internal FP32 operations; a BF16 model label does not prove that all retained sufficient factors are BF16 or that a particular reduction tree can be reproduced from them.
- Replaying must preserve the reference microbatch boundaries, per-microbatch rounding, accumulation order, any distributed reduction, and clipping/update order. Exact real-arithmetic factorization does not establish bitwise equivalence to a backend GEMM or to a different full-parameter training stack.

## Batch/sequence interpretation

The repository launch script specifies batch 1, max length 128 and accumulation four (`references/qwen27b-ascend-training/experiments/qwen27b-swift-sft/run_baseline.sh:14–16`); its logged configuration agrees (`.../evidence/2026-10-23/qwen27b-swift-sft.json:235–238`). The data preparer selects sequences with tokenized lengths **24 through 128** (`.../prepare_data.py:12–19`). Therefore 4×128=512 describes a full-length token-slot case. The saved evidence does not provide every actual per-step token length or a collator padding trace.

Those eight observed optimizer steps trained only LoRA, not all P text parameters: `train_baseline.py:13–16` asserts that every trainable name contains `lora_`, and the launch script selects LoRA at lines 11–12. Using the same sequence/batch settings for full-parameter replay is a hypothetical workload, not a previously observed full-parameter run.

N means the sum of token slots across the microbatches whose contributions form one optimizer step. If two data-parallel workers each use batch 1, accumulation four and full length 128, the global token count is 1,024, not 512. The per-device budget then depends on exactly how histories and state are sharded or replicated; dividing a global number by two is not a demonstrated implementation.

## Analytical frontier

For history horizon T, factor-byte rate f, already-aggregated small-gradient payload S, per-step metadata M, and fixed metadata/workspace R:

B(N,T) = T×(Nf+S+M)+R.

`qwen_audit.csv` and the JSON cover six N values, six horizons, and six scenarios. The explicitly hypothetical reserve scenario uses 16 bytes/step plus 16 bytes/microbatch for four microbatches, 1 MiB fixed metadata and 8 MiB workspace. **These reserves are illustrative assumptions, not measured or sufficient-for-production bounds.** Other scenarios leave these unknown costs at zero to expose the optimism rather than conceal it. In the alternative small-parameter scenario R includes its persistent m/v and full current-gradient buffer, not small-gradient history.

At N=512:

| Scenario | One step (GiB) | Four steps | Eight steps | Sixteen steps |
|---|---:|---:|---:|---:|
| BF16 factor payload alone | 6.03223 | 24.12892 | 48.25784 | 96.51569 |
| Plus all small FP32 gradients | 6.04209 | 24.16834 | 48.33669 | 96.67337 |
| Plus illustrative metadata/workspace reserve | 6.05087 | 24.17713 | 48.34547 | 96.68216 |
| FP32 factors plus small gradients | 12.07431 | 48.29725 | 96.59450 | 193.18900 |
| No input sharing, BF16 plus small gradients | 7.21396 | 28.85584 | 57.71169 | 115.42337 |
| BF16 factors, persistent small m/v and full current-small-gradient buffer | 6.06180 | 24.15849 | 48.28741 | 96.54525 |

Even the payload-only N=512 scenario allows only **four whole history steps under a 30 GiB replacement budget** or **nine under 60 GiB**. It remains smaller than the raw two FP32 moments only through **33 whole steps**. With all factors FP32 plus small gradients these thresholds become **two, four, and sixteen** steps. None of these budgets is the verified amount of free device memory.

History grows with T. Discarding old factors changes the replay unless an exact sufficient anchor is retained; a full dense m/v anchor itself costs 200.39 GiB. Thus a short-horizon saving is not a fixed-size long-run Adam representation.

## Reproduction

Run `python research/adam-state-cpu/stage11/qwen_audit.py` from the workspace. It reads the archived config and local snapshot only, requires Python's standard library, and writes JSON/CSV inventories and budget frontiers. It does not import or execute the archived Transformers source, fetch weights, install dependencies, touch devices, change the experiment core, or access the user's computer.
