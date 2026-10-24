# Stage 10 independent source and accounting audit

Audit date: 2026-10-23 UTC. Scope: the existing `factor_replay.c`, `bias_constants.h`, pilot binary/disassembly/stack report, pilot raw dumps, and the byte-identical formal rebuild. Core sources were not modified. Small independent allocation and budget probes were run on the assistant workspace only. This audit is not a GPU/NPU benchmark or a general Adam implementation certification.

## Executive assessment

The source implements genuine finite-horizon temporal replay of captured training factors. It does not replace Adam's recurrence by a low-rank approximation, read gradients from output files, hide a full moment array in the tiled path, or regenerate past factors using current weights. The old factors are captured before a step's updates, retained in full FP32, and used to reconstruct that historical gradient with the same scalar multiplication/reduction order as the reference. Bias parameters participate in capture, moment replay, weight updates, and validation.

The pilot reference, full-gradient replay, and tiled replay have identical complete factor, gradient, and state dump bytes through 16 steps. This is strong evidence for equality within the specified scalar FP32 implementation. Because the C reference and candidates share the capture/reconstruction/update functions, this alone does not independently validate the model equations. The main validation task subsequently completed a separately implemented forward/backward/optimizer oracle and external capacity tests; see the closing addendum. Those are inspected main-task evidence, not experiments claimed as this auditor's own execution.

The memory ledger includes the full reserved factor capacity, unused slots, record headers, all three work buffers, current weights, run metadata, allocator usable-size slack, a block-header allowance, numerical constants, and the supplied live-stack allowance. The formal caller/algorithm stack chain is 968 bytes. Important boundaries remain: this is an algorithm-state envelope, not RSS or a bound on every runtime allocation; `stack_bytes` is caller-supplied; validation capture uses extra allocations outside that envelope; and postallocation rejection happens after the arena itself has briefly been allocated.

The adjacent before/after hashes in the rejection branch are not, by themselves, a compelling independent atomicity test. They are redundant snapshots around only status bookkeeping. Source ordering supports rejection before capture/update, but external raw-prefix comparison, operation counters, and a zero-capacity test should establish that property empirically. Complete runs print the atomic flags as true because both untouched fields are zero, so those flags are not applicable to complete runs.

## 1. Exactness contract

### The reference actually tested

- Shape is fixed: D=32, H=128, output=16, microbatch=4, accumulation=2, P=6,288.
- Every scalar addition, multiplication, and division returns a volatile FP32 result. The supplied formal build disables fast-math and FMA contraction and uses no red zone.
- The machine code uses separate scalar `mulss`, `addss`/`subss`, `divss`, and `sqrtss` instructions, not a matrix-multiply library. No FMA was found in the inspected arithmetic paths.
- For matrix gradients, each four-sample microbatch has its own sequential FP32 sum, then the two microbatch sums are accumulated sequentially. Tiling changes output coordinates, not the K reduction order.
- Bias gradients use the same four-sample then two-group accumulation structure. Their already-reduced FP32 values are stored in each record.
- The moment recurrence restarts from positive zero in each tile and applies all historical gradients in time order. The reference initializes its persistent moments to zero and applies the same recurrence once per step. The final weight update consumes the same previous weight, final moments, correction constants, and scalar operations.
- Current weights are updated tile by tile only after the whole current factor record is captured. Reconstruction reads records, not live weights; hence an early tile update cannot contaminate later tiles' current-step gradient.
- `C1_WORDS` and `C2_WORDS` independently match FP32 rounding of Python binary64 `1.0 - beta**t`, with beta=0.9/0.999, for all t=0 through 16. Entry zero is unused by a completed update.

This does **not** establish bit identity with arbitrary BLAS/GEMM, tensor cores, TF32/BF16 kernels, fused or foreach AdamW, different K reduction trees, distributed reductions, different correction evaluation conventions, or training frameworks. The exactness statement must name the scalar FP32 reference and arithmetic order. The runtime checks round-to-nearest, but does not independently configure/check every floating-point environmental feature, such as FTZ/DAZ. The tested finite dataset has no nonfinite values; adversarial IEEE special-value behavior was not exhaustively tested.

### Pilot raw evidence

All three modes match byte for byte, rather than merely matching the implementation's rolling digests:

| Artifact | Bytes per mode | Compared contents | SHA-256 |
|---|---:|---|---|
| states | 1,207,296 | 301,824 uint32 words, w/m/v for 16×6,288 coordinates | a8d8ba665821ad5a0cc206c96a13a7be4d1d9d233b655b86f866e04b66222814 |
| gradients | 402,432 | 100,608 uint32 words | 44db4b144f2a1b070ff93eb1d79775ab6507dea52b2c03f93ca92da2620e0a0e |
| factors | 165,120 | 16 complete 10,320-byte records | 81455b3a0ea236391df68eb7e7074d19fdf06a879154b29c891e6827f6322127 |

All record headers contain consecutive steps 1–16, accumulation=2, microbatch=4, and format=0x0f320001. No nonfinite float words were present. Evidence is in `audit_pilot_evidence.json`. The hashes above identify the pilot files; they must not be substituted for hashes from other seeds or formal runs.

## 2. Non-cheating temporal storage

Each record stores both microbatches of:

- X: 2×4×32 floats = 1,024 bytes.
- hidden ReLU activations: 2×4×128 floats = 4,096 bytes.
- hidden backprop factors dZ: 2×4×128 floats = 4,096 bytes.
- output backprop factors dY: 2×4×16 floats = 512 bytes.
- aggregated bias gradients: (128+16) floats = 576 bytes.
- four uint32 metadata fields: 16 bytes.

Total numerical payload is 10,304 bytes; total record size is 10,320 bytes. The JSON field `factor_float_bytes=4` is the size of one scalar, not the record's total floating payload.

Capture computes activations and dZ/dY from the real pre-update weights. Only the new record is captured at each completed step. It neither redoes old forward/backward passes with changed weights nor treats gradients as exogenous independent random inputs. X is generated deterministically for this toy workload but then stored in the actual factor record; it is not silently replayed from an uncounted historical input cache. The current weight array is always present and is charged separately. The generated initial weights are toy initial conditions, not a concealed pretrained-weight copy.

In mode 2, the work area has only B floats each for m, v, and g. No full P-length m, v, or gradient array exists in that path. A local `State` is only one coordinate, accounted for by the compiled frame. `Proof` is NULL during timed/charged execution, so its full histories are not read or allocated by the algorithm. In mode 1, a full gradient buffer is deliberately present and charged; it is not equivalent to the tiled storage claim.

There is no compression or eviction of factor history and no mechanism for indefinite operation under fixed capacity. Step capacity+1 stops the run. This is a finite-horizon time/memory tradeoff, not a fixed-size lossless state codec or a solved long-run optimizer replacement.

## 3. Memory ledger and formulas

Observed ABI sizes are Run=136 bytes, Constants=168 bytes, Record=10,320 bytes, State=12 bytes. Current weights use 4P=25,152 bytes; raw FP32 m/v use 8P=50,304 bytes.

Let C be reserved history capacity, B tile size, S the derived live stack allowance, and delta=malloc_usable_size−requested_bytes. The reported optimizer replacement subtracts only the common current-weight payload from the full algorithm charge:

- Mode 0 reference: 86,224 + S + delta bytes.
- Mode 1 full-gradient replay: 10,320C + 8B + 25,600 + S + delta bytes.
- Mode 2 tiled replay: 10,320C + 12B + 448 + S + delta bytes.

The constant 448 in the tiled formula is 136 run metadata + 168 numerical constants + 128 extra constant-pool reserve + 16 allocator-block allowance. It does not include weights, the full record allocation, or workspace, which are explicitly separate terms. The full algorithm envelope adds 25,152 bytes back to these optimizer figures.

The requested history array is based on capacity, not completed steps. A step-1 run with C=16 therefore charges all 16 slots, including 154,800 bytes of unused history. The full current record is not double-counted outside that array. Reference mode reuses one record, rather than retaining 16 records.

### Actual allocator rounding

Forty-three small probes covered reference/full/tiled modes, capacities 0/1/2/4/5/8/16 as applicable, and B=1/256/6,288. The returned usable allocation was recorded, not assumed equal to the request. Some larger blocks included substantial mmap/page rounding. For example, tiled B=256/C=16 requested 193,480 bytes and reported 196,592 usable bytes: 3,112 extra bytes, all charged. The largest observed excess among these probes was 3,560 bytes. Small B=1 tile allocations had four bytes of usable-size excess. These are observations for this allocator/process configuration, not portable allocator guarantees.

`audit_memory_probe.json` stores these raw probes with S=1,024, deliberately a conservative allowance above 968. Do not mix their reported totals directly with the formal S=968 table: formal values with the same allocation differ by 56 bytes.

### Constants and stack

The binary contains C as a 168-byte object. Inspected direct numerical constant-pool references outside C occupy approximately 68 bytes, including the double loss-scaling constant, so the separate 128-byte numerical reserve is conservative for this build. Format strings, code bytes, ELF metadata and shared-library mappings are runtime/infrastructure exclusions, not optimizer state.

The formal `.su` file reports main=512, algorithm=272, capture=184, reconstruct=80, update_weight=56, validate_dump=944 bytes. The maximum included caller chain is 512+272+max(184,80,56)=968 bytes, not merely the largest single frame. `-mno-red-zone` is present in the formal compile command. Main's frame is dynamic but bounded; using its reported maximum throughout is conservative. The proof/dump chain is larger and intentionally outside the charged algorithm path.

The compiled algorithm does call libc `memset` for clearing moments. Its validation-only path can call `memcpy`. The math update uses an inline hardware sqrt instruction. A report must not claim the algorithm has no external calls. The 968-byte sum covers the named application frames, not the innards of libc routines, the loader, initial stack/argv/environment storage, signal machinery, allocator arenas/tcache, stdio, timing, or operating-system mappings. The 16-byte block allowance is not a charge for the whole allocator arena. This is therefore not a whole-process peak-physical-memory bound.

### Finite-horizon crossover

Using the formal S=968 allowance and the observed small-block rounding, a completed C=1/B=1 tiled case charges 11,752 optimizer bytes, or approximately 23.36% of raw FP32 m/v. At C=4/B=256 the same formula gives 45,768 bytes, approximately 90.98% of raw m/v. At C=5, history alone is 51,600 bytes, already above raw m/v=50,304, regardless of B or stack.

Thus there can be genuine net replacement-memory savings in this toy shape for short history, and no below-raw-m/v result for five or more retained steps. A ratio based on the larger practical reference workspace is a different comparison and must not be presented as the raw-moment ratio. None of these values establish a 30/200.39 memory target or a scale-up result for a transformer.

## 4. Budget screening strengths and limitations

Preflight uses the requested payload plus a 32-byte allocation margin, numerical constants and supplied stack. After `calloc`, it checks the actual usable-size-based charge plus a 16-byte block allowance. Both checks precede algorithm initialization/capture/update.

An independent formal probe with mode=2, C=16, B=256, S=968 and budget=170,000 correctly returns `budget_rejected_after_allocation`. This exercises a case where nominal preflight fits but page-rounded usable bytes do not. The block is freed and the algorithm is not started. However, the allocation has already happened briefly, so the mechanism does not prove that a hard physical-memory ceiling was never crossed.

Preflight can also reject conservatively. For mode=2, C=1, B=1, S=968, its estimate is 11,764 bytes, while the observed completed charge is 11,752. A budget of 11,752 is rejected before allocation; a budget of 11,764 is accepted. This small gap is safe, but a rejected budget boundary is not proof of mathematical infeasibility.

Most importantly, S is trusted command-line input, not measured/enforced by the executable. Supplying S=0 lets the same C=1/B=1 case pass an 11,000-byte budget while undercounting the live application stack. The controlled formal driver must derive and always supply 968 (or a larger documented bound). This is an experiment protocol, not an adversarial input-safe resource limiter. Robust productionization would require validated numeric parsing, checked integer addition and an enforced build-specific stack lower bound; untrusted huge `strtoull` values can also overflow the unsigned accounting arithmetic.

Raw budget cases are in `audit_budget_probes.json`. No core-source changes were made to address these limitations; externally fixed inputs suffice for the controlled study if the limitations are disclosed.

## 5. Capacity rejection and atomicity

The capacity check appears at the start of the timestep, before choosing/capturing a new record, generating that step's inputs, reconstructing gradients, replaying moments or changing weights. For C=0 it rejects the first attempted update after initial weights have been initialized. It is not a claim that the entire arena remains identical to an all-zero pre-initialization allocation.

The rejection branch hashes weights/history, changes status/rejected_step, then hashes the same arrays again. Those flags can detect accidental writes inserted into that short branch, but do not independently demonstrate that all prior code paths avoided starting the next step. Hash collisions are another reason to prefer full byte comparisons for this small test.

Recommended external criterion:

1. For capacity C, compare a request for C+1 or more steps against a fresh run stopped after exactly C steps with the same seed/capacity/block.
2. Compare every dumped gradient/state/factor byte for the completed prefix and exact lengths; compare digests only as additional diagnostics.
3. Require completed=C, rejected_step=C+1, capture_calls=C, and the expected replay/reconstruction counters. There must be no evidence of a partial extra capture or moment transition.
4. Include C=0, C=1, a nondivisor tile size, and capacities adjacent to the maximum supported 16.
5. Interpret atomic flags only when status is `capacity_rejected`; true on `complete` is merely the zero-initialized default.

This is atomic rejection of a known capacity condition, not a general transactional API. The implementation does not promise rollback after arbitrary faults midway through an otherwise admitted step, resume from a saved prefix, accept externally loaded records, or validate corrupted record metadata.

## 6. Validation phase and I/O exclusions

`validate_dump` runs after the timed loop, allocates full states, gradients and factors for requested steps, zeroes the original arena, reruns cold, and checks status/completed/digests against the timed run. This checks reproducibility without feeding saved gradients back into the measured path.

For T=16 the extra requested proof payload is 1,207,296 + 402,432 + 165,120 = 1,774,848 bytes before allocator overhead, stack and stdio. It cannot be described as part of a low-memory execution. It is valid to keep it as a separate verification phase if reported explicitly. Running the executable with a non-"-" dump prefix does not constrain the whole process to the printed optimizer budget.

The dump writer checks `fopen` but not return counts of `fwrite` or errors from `fclose`. External byte-count and full-content comparisons are therefore important. A hardened future writer should check short writes and propagate I/O errors; no change is needed to the arithmetic core for that improvement.

## 7. Prioritized further tests

The main validation task has already stated plans to perform several of these; this list is not a claim they all finished:

1. Independently implemented full FP32 forward/backward/AdamW oracle, comparing raw factors, gradients and w/m/v, not just final loss.
2. Multiple seeds; all steps through 16; B=1, nondivisors, boundaries near bias transitions, and B=P. Explicitly compare all 144 bias coordinates.
3. External raw-prefix capacity tests above, including zero capacity and over-reserved capacity.
4. Budget grids around both nominal and actual charged thresholds; retain rejected rows. Include a verified postallocation rejection caused by allocator rounding.
5. Rebuild reproducibility with captured compiler command, source/header/binary hashes, `.su`, platform, and disassembly. Recompute stack/constant bounds if the binary changes.
6. Memory safety checking with already-available tooling only, or sentinels around the requested arena and workspace, in a separate uncharged verification binary. Distinguish sanitizer runtime memory from production memory.
7. Verify finite outputs and exact artifact lengths before accepting equality; do not treat matching NaN-derived losses or two truncated empty files as success.
8. If generality is later claimed, separately test signed zeros, subnormal arithmetic, extreme magnitudes, nonfinite propagation, rounding-mode rejection, and FP environment assumptions. Those are beyond the present toy-data evidence.

## 8. Source identity

The formal source/header copies were byte-identical to the audited originals at inspection:

- `factor_replay.c`: SHA-256 0766be5147383ee650af391fc3a2483cc82f89702d05097f8e1353f0ba62df2b.
- `bias_constants.h`: SHA-256 75257260a6e237cf3b69923bdcf22e4a4dda38a94d663b32bb68f42e961f0bbe.

The formal compiler was GCC 14.2.0, glibc 2.41, x86_64. Its recorded command includes `-O2 -fno-fast-math -ffp-contract=off -fno-math-errno -mno-red-zone -fstack-usage -Wall -Wextra -frecord-gcc-switches`. Results remain conditional on this reference and the documented accounting exclusions.

## Closing addendum: formal evidence available at audit finalization

The auditor inspected `formal_results/summary.json`, `oracle_parity.json`, `capacity_arena_equality.jsonl`, the external `capacity_harness.c`, and the validation driver. They record:

- 25 references and 175 completed replay candidates, with 30,089,220 compared raw uint32 words including record metadata, all equal.
- Five separately implemented NumPy full-trajectory oracle matches at seeds 0, 1, 42, 2026 and 2,147,483,647: 2,218,560 compared words.
- 24 rejected-run raw prefixes and 24 independent stop-versus-reject payload comparisons covering capacities 0/1/2/4/8/15, two seeds, and both replay modes. All payload comparison records parse and report pass. The harness directly compares weight, entire history, and entire workspace bytes plus operation counters and digests. It deliberately does not require the Run headers to be byte-identical, because requested steps, rejection status and rejection snapshots differ.
- 134 budget checks: 88 rejected and 46 completed. None accepted the 10%-of-raw-m/v budget. The driver computes the 968-byte application-stack allowance from the formal `.su`, rather than using an unchecked caller-selected zero.

These address the shared-reference-only and adjacent-hash-only evidence gaps identified above for the tested cases. They do not broaden the scalar exactness scope, finite-horizon limit, or memory-accounting exclusions.

The auditor also verified the successful rerun's `capacity_address_undefined.jsonl`: 24 valid records, all passing; `capacity_address_undefined.stderr` is empty. The main task reports exit code zero. This is the clean AddressSanitizer+UndefinedBehaviorSanitizer evidence, with unsupported leak detection disabled. Separately, the original `capacity_sanitized.jsonl` contains 23 valid records followed by a truncated 24th record, and `capacity_sanitizer.stderr` explicitly reports LeakSanitizer's unsupported ptrace environment. Those original files are intentionally retained failure-attempt logs, not the passing rerun. LeakSanitizer remains unverified; no address/undefined-behavior violation was reported in the passing rerun.
