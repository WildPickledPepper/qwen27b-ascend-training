# Independent Stage13 residual-codec audit

## Verdict

All **152 saved frames** pass an independent decoder and full-file accounting audit: 132 frames from 12 genuine trained-state snapshots, plus 20 constructed/edge controls. All **20,661,088 decoded bytes** match the source bytes. No correctness or missing-cost defect was found in this bounded experiment.

**None of the 132 trained-state frames meets the full-frame ≤14.97074684% target.** Within Stage13's 11-method menu, plain byte-shuffle + zlib beats the BF16 and rank1/2/4/8 predictor-residual candidates on every trained snapshot. At Stage5 step4096 the best full frames remain 82.9436%–83.0772% of raw m/v; Stage10 step16 is 85.3133%.

Constructed known-rank2 controls demonstrate that the coding mechanism can work when the exact predictor or sparse corrections are truly available: complete frames reach 3.2959% and 4.8164%. Making low16 corrections dense raises the best control to 53.3897%. These are synthetic state constructions, **not trained Adam trajectories**, and do not supply a large-model result.

The conclusion is limited to these data and these codecs. It is not a universal lower bound, an impossibility proof, or an online-memory certificate. “Baseline wins” describes this Stage13 menu only: earlier Stage5 fitted m²→v predictors were better on first-step states and were not rerun here.

## Scope and acceptance criterion

This audit does not run training or modify the main experiment. It reads existing genuine Stage5 and Stage10 FP32 Adam snapshots, independently checks mathematical reversibility, and verifies every saved main-experiment frame and its full byte accounting. Code uses only existing NumPy and Python standard library. No installation, GitHub modification, remote/user computer, NPU, or state offload is involved.

The acceptance denominator is raw dense FP32 m/v, including bias states, with a target full-frame ratio at most 0.1497074684. It is not residual-only cost, and it is not a ratio to a metadata-inflated baseline. Thus:

- Stage5 raw m/v = 172,064 B, requiring a complete frame of at most 25,759 B.
- Stage10 raw m/v = 50,304 B, requiring a complete frame of at most 7,530 B.

A passing offline-storage size alone is not a 30 GiB online optimizer peak-memory result. Encoder factors/SVD workspace, decoder workspace, old/new frame coexistence, indices, buffers, worst-case fallback, allocator overhead and device implementation require separate accounting and validation.

## Provenance checked without rerunning training

`independent_verify.py` verifies all 21 Stage5 snapshots, not just the nine selected for this experiment. All 3,613,344 bytes match the recorded per-file SHA256 and exact raw sizes. The 21 entries are the full three-seed × seven-step grid. Collector, teacher/student code and Adam source hashes match the original run metadata. All saved states are finite and v is nonnegative. The metadata and source show a 16→1024→4 tanh network, 21,508 active parameters, ordinary FP32 AdamW states and no included padding coordinates.

For Stage10, the existing seed42 16-step C reference states match its separately implemented NumPy oracle byte-for-byte (1,207,296 B). Both state files, the parity report and C source match the Stage10 manifest. The audit independently extracts steps 1, 8 and 16 from the interleaved [weight,m,v] trajectory, retaining all 6,288 parameters. This includes 144 biases in each of m and v, totaling 1,152 raw bias-state bytes. No trajectory was recomputed.

These checks establish consistency with saved provenance. They are not an independent re-creation of the historical training run, or evidence of large-model state behavior.

## Independent ordered-word and residual checks

The auditor uses its own scalar integer specification, vectorized implementation and inverse. For a 32-bit IEEE word w:

- Negative-sign words map to 0xffffffff − w.
- Nonnegative-sign words map to w + 0x80000000.
- The two disjoint branches map bijectively to the low and high half of uint32. The inverse reverses the corresponding branch.

This proves the transform is a bijection for every 32-bit word, including both zeros and all NaN payloads. Finite values sort monotonically, and −0/+0 occupy adjacent distinct ranks. Differences involving infinities or NaNs are coding quantities, not numerical ULP-distance claims.

The automated test covers 655,382 words: every high16 pattern with six low16 boundaries, 262,144 seeded arbitrary words and explicit IEEE edges. It then checks 2,621,530 predictor pairs, including random predictors, BF16-truncated predictors, zero predictors and rank extremes. Raw-word inverse, XOR residual inverse, signed ordered-rank delta inverse, ZigZag inverse, and signed-zero separation all pass. The set includes 2,567 subnormal and 2,527 NaN cases, positive/negative infinity, and signaling/quiet NaN payload representatives. This is a bounded test, not an exhaustive enumeration of all words.

Signed ordered-rank residuals span [−4,294,967,295, +4,294,967,295]. ZigZag therefore reaches 8,589,934,590 and requires up to 33 bits. A signed-int32 difference would be insufficient unless the format explicitly used a different modular representation. The independent tests use int64 differences and uint64 ZigZag.

## Independent full-frame verification

`independent_frame_verify.py` parses every saved file without importing the main codec or calling its decoder. It independently reads the 28-byte header and JSON, validates exact file length, unshuffles bytes with strided bytearray assignments, decodes varints with its own integer parser, and reconstructs predictors row-first. Its decompressor also rejects unused trailing stream data. The raw input is consulted only after decoding to verify bitwise equality; it is never an input to predictor reconstruction.

Every full frame is exactly:

28-byte header + JSON metadata + compressed predictor stream + compressed residual stream.

All 152 files sum to 20,201,171 B. Every CSV/JSON component length and ratio matches actual file bytes. Source identity, shapes, step, factor ranks, vector/bias data, and whole-state SHA256 match. All 12 selected inputs were independently compared with original Stage5 snapshots or extracted from the original Stage10 interleaved trajectory. Main source and input-index hashes also match its run record.

There is no missing anchor, oracle, predictor dictionary or external factor dependency. Compressed factors and BF16 high16 predictors are in each frame. Decompressed predictor lengths agree with the independent formula below. All bias vectors are exact before correction, and their storage is charged.

The verified tensorwise low-rank raw predictor payload sizes, including exact bias vectors, are:

| Dataset | rank1 | rank2 | rank4 | rank8 |
|---|---:|---:|---:|---:|
| Stage5 | 24,768 | 41,312 | 74,400 | 107,680 |
| Stage10 | 3,584 | 6,016 | 10,880 | 20,608 |

Ranks are clamped to tensor dimensions, e.g. the Stage5 1024×4 matrix has at most four columns. Payloads are compressed and every resulting predictor-stream byte is charged. BF16 predictor payloads are 86,032 B for Stage5 and 25,152 B for Stage10 before their own lossless compression.

The independent decoder reconstructs explicit FP32 multiply-then-add outer products rather than silently using BLAS reduction order. It additionally verifies 1,980 selected output entries using scalar FP32 multiply and add rounding. All pass. The direct known-factor source check adds ten scalar checks. SVD is encoder-only, and the stored FP32 factors, not SVD implementation details, determine decoding. This success is scoped to the recorded CPU/NumPy semantics, not certified NPU or cross-platform arithmetic.

## Real-state results and bounded interpretation

| State sample | Best Stage13 full-frame ratio | Winning method |
|---|---:|---|
| Stage5 step1, three seeds | 86.4963%–86.6056% | Byte-shuffle + zlib |
| Stage5 step128, three seeds | 84.2158%–84.3355% | Byte-shuffle + zlib |
| Stage5 step4096, three seeds | 82.9436%–83.0772% | Byte-shuffle + zlib |
| Stage10 seed42 step1 | 88.7146% | Byte-shuffle + zlib |
| Stage10 seed42 step8 | 86.1224% | Byte-shuffle + zlib |
| Stage10 seed42 step16 | 85.3133% | Byte-shuffle + zlib |

Across the selected Stage5 snapshots, BF16+XOR is 83.3684%–87.0792%; rank1+XOR is 90.0182%–96.6838%. Larger factors generally increase full-frame cost. Stage10 rank8 benefits from its unusually short first-step structure but still does not beat that snapshot's baseline. Signed-rank varints are not automatically better than XOR/shuffle when residuals are broad.

These comparisons charge the whole frame against raw m/v. An accurate low-rank numerical approximation is not enough: the exact correction stream and the predictor both cost bytes. Nor do marginal low-bit entropies establish joint incompressibility. No search over all possible predictive structures was attempted.

## Controls, special values and rejection paths

All 20 saved controls independently decode to exact source bytes. All six low-rank constructed frames contain the same 4,096 raw factor bytes, individually compressed to 3,708 B and charged in each frame.

| Constructed source | Best complete frame | Header/metadata | Predictor | Residual | Ratio |
|---|---:|---:|---:|---:|---:|
| Exact known rank2 | 4,320 B | 560 B | 3,708 B | 52 B | 3.2959% |
| Sparse ±1 ordered-rank corrections | 6,313 B | 565 B | 3,708 B | 2,040 B | 4.8164% |
| Dense random low16 | 69,979 B | 558 B | 3,708 B | 65,713 B | 53.3897% |

The exact case matches the auditor's own factor reconstruction. The sparse case has precisely 1,638 changed words out of 32,768, each by +1 or −1 ordered rank. This is floor(5% × count), or 4.9987793%, not exactly 5%. The dense case keeps every original high16 word and changes 32,767 of 32,768 low16 words. The target is met by four control frames (both residual formats on the exact and sparse sources), not by any trained frame.

All five saved edge frames preserve the 32 raw IEEE words, including −0, smallest/largest subnormals, finite extremes, both infinities, and different quiet/signaling NaN payloads. The best edge frame is 460 B for 128 raw bytes, so fixed metadata dominates this intentionally tiny case.

`independent_behavior_verify.py` additionally compares the production ordered transform with the independent specification on 655,360 words, validates 10,012 varints, and independently decodes six in-memory special frames. The extra low-rank cases put nonfinite payloads only in exact vector blocks: payloads remain bit-exact while the SVD matrix inputs are finite. A nonfinite matrix is deliberately rejected by SVD; baseline, zero and BF16 predictors can still encode its raw words.

Malformed/truncated/too-wide/out-of-range varints and wrong counts are rejected. Sixteen modified frames (four modifications across four predictor/residual pairs: truncation, append, magic and payload corruption) are rejected. These are bounded robustness checks, not a hostile-input security audit.

## Remaining limits

- No online optimizer update loop, full training replay, convergence test, throughput advantage or device implementation was added here.
- The encoder temporarily holds dense source states and SVD arrays. Decode also constructs full arrays. The files' compression ratios are not peak-memory measurements.
- The synthetic known factors come from the controlled construction, not from a successful trained-state factor-discovery algorithm.
- Neither finite-data tests nor special-value tests certify NPU GEMM, fused arithmetic, FTZ/DAZ, alternate rounding modes, arbitrary NaN arithmetic or cross-platform low-rank reconstruction.
- SHA256 and zlib checks detect tested corruption but do not provide authenticity against a malicious sender.

## Reproduction and standalone archive validation

The frame and behavior auditors work from the standalone Stage13 directory, without Stage5/Stage10 siblings:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python independent_frame_verify.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python independent_behavior_verify.py
```

The frame auditor uses stronger original-file comparisons when sibling Stage5/Stage10 files are present. If they are absent, it explicitly selects these archived-evidence routes and records the choice per input in `audit_frame_results.json`:

- Stage5: compare each archived input's SHA256, byte length, seed, step and shapes with `provenance/stage5_snapshots_index.json`.
- Stage10: verify the SHA256 of `provenance/s42_t16_m0_b128_c16.states` against the original trajectory hash in the input record, then independently extract each selected m/v state, including biases.

This fallback verifies the preserved evidence and all frame decoding, not the original training process. No training is rerun or independently re-created.

The fallback was exercised using `audit_standalone_verify.py`, which made a temporary copy containing the required scripts and inputs/provenance/results directories, verified that Stage5 and Stage10 siblings did not exist, and ran only the independent frame auditor. All 152 frames passed with the same hashes, bytes, decoded states, controls and accounting as the original-location audit. The evidence routes were nine archived Stage5-index checks, three archived Stage10-trajectory extractions and four control inputs. The preserved log is `audit_standalone_verification.log`; machine-readable results are `audit_standalone_results.json`. This copy test can itself be repeated:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python audit_standalone_verify.py
```

The separate full-original-provenance auditor intentionally still requires sibling Stage5 and Stage10 directories. It checks all 21 original Stage5 snapshots and Stage10's original independent-oracle files, which are not all in the standalone archive:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python independent_verify.py
```

Its previously completed results remain in `audit_independent_results.json`. Other audit outputs are `audit_frame_results.json` and `audit_behavior_results.json`. The provenance and frame auditors do not import the main codec. The behavior auditor imports it only for focused dynamic checks and does not run or modify its experiment. Main codec SHA256 audited: `b152f92f0a000b4efd723bbe3921fb8794abe749cc9e789fd6c4d296770ce5ea`.
