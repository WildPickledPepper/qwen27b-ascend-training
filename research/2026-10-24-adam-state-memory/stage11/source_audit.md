# Independent Stage11 source-scope audit

This is a read-only inspection of the Stage11 changes against `../stage10/factor_replay.c`, not a new execution campaign. Core files were not modified by this audit.

## Confirmed scope

- `width_replay.c:36–48` changes only the stored input X type: `uint16_t` when MIXED_X is set, otherwise FP32. Activations, dZ, dY and already-aggregated bias gradients remain FP32 in both configurations.
- `width_replay.c:87–90` constructs each source datum directly as a BF16 bit pattern, then lifts it to FP32. Both storage modes use this identical generator. At lines 73–85 the uint16 store drops only guaranteed-zero low bits and the load restores them exactly. This is lossless for the generated data, **not a general lossless FP32-to-BF16 conversion**. It also changes the task's dataset relative to Stage10's signed-unit FP32 generator, so the new stage should not be described as a byte-identical continuation of the old Stage10 trajectory.
- `width_replay.c:46–48,96–100` groups all microbatches' X, then all activations, dZ, dY, and one bias vector. This replaces Stage10's per-microbatch interleaving. It is a layout change, not factor quantization.
- `width_replay.c:103–134` still captures factors at current historical weights before any parameter update. Forward/backward and bias accumulation stay in the explicit FP32 arithmetic helpers; X is loaded losslessly where used.
- `width_replay.c:138–152` changes matrix-gradient K traversal only in mode 3: sample indices reverse inside each microbatch. Microbatch-group order remains ascending. Already-aggregated bias gradients bypass this change. `ACC=2` is unchanged. For MB=1 the reversal is the identity; with MB>1 it is a deliberate reduction-order negative control, not a precision change.
- `width_replay.c:167–199` retains the same finite-history reconstruction strategy. Modes 2 and 3 use moment and gradient tiles, replaying all stored historical gradients from zero moments. Mode 0 holds dense m/v/g. The new mixed-width record does not reduce FP32 m/v precision.
- `run_frontier.py` reads each compiled binary's `.su` file and charges `main + algorithm + max(capture,reconstruct,update_weight)`. Checked sample `.su` files contain those measured frames, with mixed-input capture frames sometimes larger than all-FP32 frames. The protocol does not reuse one stage-wide stack constant. It still excludes the separate raw-proof validation allocation/frame from the algorithm envelope, as it should explicitly disclose.

## Record and memory formula

For N=ACC×MB and arbitrary D/H/K, the record payload plus header is:

16 + N×D×sizeof(XFactor) + 4×[N×(2H+K)+(H+K)] bytes,

plus any compiler struct padding (the grid's asserted `sizeof(Record)` values verify no extra padding for those compiled shapes). For D=H=K=n this becomes 16+Nn×(sizeof(XFactor)+12)+8n. Hence mixed X saves exactly **2Nn bytes per history step** for these shapes. It does not halve the entire record. The run header, retained capacity, current weights, per-binary stack, constants and allocator usable capacity remain separate charges (`width_replay.c:220–225`).

The CLI `factor_float_bytes=4` refers to the remaining original FP32 factors, not to every field in a mixed record; `native_input_storage_bits` and `original_fp32_factor_bytes_per_step` disambiguate it (`width_replay.c:231–232`).

## Minor archive-format caveat

`FORMAT_VERSION` remains `0xF320001` at `width_replay.c:33`, despite both the changed grouping and the two possible X storage types. The 16-byte record header alone does not identify dimensions, dtype, or the changed layout. The raw `.factors` files must therefore be read together with the archived build flags and run protocol; the protocol's decoder does exactly that. This is **not a numerical-parity failure**, but the format is not self-describing and should not be advertised as a standalone interoperable serialization format.

## Boundary to the Qwen analysis

The exact preservation of native BF16 **inputs** in this scalar CPU MLP does not validate storing Qwen's entire factor set in BF16. Stage11 retains original FP32 activations and derivatives; the Qwen budget's all-factor BF16 case is intentionally optimistic and remains unimplemented. Likewise, the scalar-order negative control supports the need to preserve a specified reduction order, not a claim that arbitrary GEMM/NPU kernels can be reproduced bitwise by this source.
