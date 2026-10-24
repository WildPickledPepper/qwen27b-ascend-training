# Stage 12 independent algebra audit

## Bottom line

Historical gradient factors admit exact **real-arithmetic** concatenations for the first moment and a symmetric Hadamard-square lift for the second moment. Neither identity, by itself, preserves the declared sequential FP32 Adam trajectory. There are two independent rounding boundaries: constructing the gradient from factors and advancing the moments. There is also a constants boundary: the baseline's separately stored injection constants are not exactly `1 - stored_beta`.

A four-scalar counterexample is enough to reject once-rounded algebraic consolidation as a general bitwise replacement for this baseline. This does **not** reject all exact compression or all exact recomputation schemes. In particular, replaying the original ordered FP32 operations remains a distinct, valid candidate.

Scope: audit of `stage11/width_replay.c` and `stage11/independent_oracle.py`, exact derivation, and tiny independent rational checks. No training, large matrix run, install, GitHub write, device/offload experiment, or hardware performance claim is part of this audit.

## 1. State exactly which arithmetic is being modeled

Let a weight matrix have shape `d × e`. At update `s`, its accumulated token-factor columns are

`A_s ∈ R^(d × r_s)`, `B_s ∈ R^(e × r_s)`.

The ideal real gradient is `G_s = A_s B_s^T`. Here `r_s` counts all token contributions entering one optimizer update, including all its accumulated microbatches. Scaling may be absorbed into either factor. The actual FP32 reference uses its declared product rounding, within-microbatch reduction, and accumulation of microbatch results; write its output as `g_s = Reconstruct32(A_s, B_s, order_s)`. In general, `g_s != G_s` as exact real matrices. Concatenating token columns is an exact-real identity, not permission to change this reduction tree.

For general constants `b1, b2, c1, c2`, define the *real* recurrence

`M_t = b1 M_(t-1) + c1 G_t`

`V_t = b2 V_(t-1) + c2 (G_t ⊙ G_t)`.

Here `c1/c2` denote injection constants only, not Adam bias-correction arrays. For a mathematically specified Adam one often sets `c = 1-b`. For this experiment one must instead use the exact real values of its separately stored FP32 constants:

| Constant | Stored binary32 word | Exact decimal value |
|---|---|---:|
| `b1` | `0x3f666666` | 0.89999997615814208984375 |
| `c1` | `0x3dcccccd` | 0.100000001490116119384765625 |
| `1-b1` | `0x3dccccd0` | 0.10000002384185791015625 |
| `b2` | `0x3f7fbe77` | 0.999000012874603271484375 |
| `c2` | `0x3a83126f` | 0.001000000047497451305389404296875 |
| `1-b2` | `0x3a831200` | 0.000999987125396728515625 |

The exact sums satisfy `b1+c1-1 = -3/2^27` and `b2+c2-1 = 111/2^33`. Replacing `c` with `1-b`, decimal 0.1/0.001, or a power computed by a different arithmetic changes the model before any consolidation issue is tested.

The baseline's update is specifically separate rounded operations:

`m_t = RN32(RN32(b1*m_(t-1)) + RN32(c1*g_t))`

`v_t = RN32(RN32(b2*v_(t-1)) + RN32(c2*RN32(g_t*g_t)))`.

An FMA variant is a different reference unless separately proved equivalent. Bias correction, square root, epsilon, weight decay, and parameter update ordering must also remain unchanged for full-trajectory equality.

## 2. First-moment concatenation over the reals

For zero initial state, let `α_s = c1 b1^(t-s)` and `R = Σ_s r_s`. Then

`M_t = Σ_s α_s A_s B_s^T = U_m V_m^T`,

`U_m = [α_1 A_1 | ... | α_t A_t]`,

`V_m = [B_1 | ... | B_t]`.

Scaling only one side avoids unnecessary square roots. A balanced `sqrt(α_s)` representation is also a real identity for nonnegative weights, but introduces another rounding choice in an implementation. The exact algebraic rank obeys

`rank(M_t) ≤ min(d, e, Σ_s rank(G_s)) ≤ min(d, e, R)`.

These are upper bounds, not predicted ranks or demonstrated storage reductions. Dependence and cancellation can lower the actual rank. A nonzero checkpoint adds the term `b1^t M_0`; its storage and rank cannot be omitted.

## 3. Second-moment lift: preserve all within-update cross terms

Write a single step as `G = Σ_(k=1)^r a_k b_k^T`. Its elementwise square is

`G ⊙ G = Σ_k (a_k⊙a_k)(b_k⊙b_k)^T`

`          + 2 Σ_(k<l) (a_k⊙a_l)(b_k⊙b_l)^T`.

Define `q = r(r+1)/2` columns indexed by `k≤l`:

`L(A)_(k,l) = a_k⊙a_l`,

`D(B)_(k,l) = (2 if k<l else 1) (b_k⊙b_l)`.

Then `G⊙G = L(A) D(B)^T` exactly over the reals. This asymmetric factor-of-two convention avoids `sqrt(2)` entirely. The alternative full ordered-pair lift has `r²` columns and duplicates the off-diagonal pairs.

With `η_s = c2 b2^(t-s)`, `q_s = r_s(r_s+1)/2`, and `Q = Σ_s q_s`,

`V_t = U_v V_v^T`,

`U_v = [η_1 L(A_1) | ... | η_t L(A_t)]`,

`V_v = [D(B_1) | ... | D(B_t)]`.

Thus `rank(V_t) ≤ min(d, e, Q)`. If `ρ_s = rank(G_s)` is an independently established exact-real rank and an exact rank-`ρ_s` factorization is available, a tighter structural upper bound is `min(d, e, Σ_s ρ_s(ρ_s+1)/2)`. This is not a claim that a floating-point rank-`ρ_s` factorization preserves the baseline.

Important boundaries:

- Cross terms between **different tokens or microbatches in the same update** are required. Squaring each microbatch gradient and then adding is not the square of the accumulated gradient.
- There are **no cross-step terms** in `Σ_s η_s (G_s⊙G_s)`. Squaring one concatenated historical sum introduces wrong cross-step terms.
- Two scalar token contributions `1,1` give aggregate square `4`, not the diagonal-only `2`. Conversely, scalar contributions `1,-1` give aggregate square `0`, not `2`.
- The off-diagonal lifted columns may contain signed values; entrywise nonnegativity of the completed square does not make every separable cross term nonnegative.
- Elementwise second moment is not `G G^T`, `G^T G`, a row variance, or a column variance. Substituting any of these changes Adam.
- A nonzero second-moment checkpoint adds `b2^t V_0` and must be counted.

## 4. What would actually be stored?

For one `d × e` matrix and one scalar storage size `s` bytes:

| Representation | Scalar payload, before metadata and workspaces |
|---|---:|
| Original factor history, shared by both reconstructions | `(d+e) R` |
| Explicit concatenated first-moment factors | `(d+e) R` |
| Explicit symmetric second-moment lift | `(d+e) Q` |
| Both separately materialized moment factorizations | `(d+e)(R+Q)` |
| Dense first and second moments | `2de` |
| Independently compressed exact-rank factors | `(d+e)(k_m+k_v)` |

Multiply by `s` only if each stored scalar truly fits that dtype while meeting the specified equality. Exact rational consolidation can increase numerator/denominator bit lengths; assuming four bytes for exact powers, factors, or transformed bases is not justified by the real identity. Mixed dtypes require separate per-factor accounting. Bias and non-matrix parameters require their own states/history and must not disappear from totals.

For equal `r_s=r`, explicit lift has `Q=t r(r+1)/2` rather than `R=tr`; it can be much larger than the original factor log. Storing the original `A/B` and generating lift terms on demand avoids storing `Q` columns, but does not consolidate away history. It trades storage for reconstruction work. Reconstructing `G` first and squaring it is another option; to reproduce the FP32 reference it must use the original rounding order.

For square `n×n` matrices, one rank-`k` `UV^T` payload is `2nk` scalars versus `n²` dense, so the payload break-even is `k<n/2`, not merely `k<n`. With two moment matrices it is `k_m+k_v<n` for separately stored factors. For a rectangular matrix the per-moment threshold is `k<de/(d+e)`. Near full rank, choosing dense storage can be cheaper and simpler. A dense fallback accurately counts `2de` state words; it does not satisfy a strict ten-percent-of-dense-state budget merely because it enables discarding history.

An adaptive implementation must also charge conversion/merge peaks, old-plus-new factors during a merge, rank workspace, dense fallback memory, metadata, bias state, and tile buffers. This audit's Python scratch objects and factor formulas are not a certified memory-bound implementation.

### Gram/core alternatives

The row lift corresponds to symmetric outer-product features: `(a_i·b_j)^2 = <a_i a_i^T, b_j b_j^T>_F`. One can store/generate these features implicitly or use bases with a core, `U C V^T`, whose payload is `d k_u + e k_v + k_u k_v` scalars. Such forms can exploit actual dependence, but do not guarantee it.

Small Grams such as `U^T U` and `V^T V` can help determine spectra when combined with the directional factors. They cannot alone reconstruct all matrix entries, because they discard ambient orientation. Lifted Grams can themselves require quadratic space in `Q`. A floating-point eigensolver/SVD with a truncation tolerance measures numerical rank and may discard nonzero directions; it is not an exact-lossless certificate. Even a full computed QR/SVD factorization changes operation order and must separately pass bitwise reconstruction tests.

## 5. Two independent obstacles to trajectory equality

### A. Matrix-product rounding can increase exact rank

Let `ε=2^-23` and `a=b=[1,1+ε]^T`, all exactly binary32. The real outer product has rank one. Rounding each product to binary32 gives

`[[1, 1+ε], [1+ε, 1+2ε]]`.

As an exact rational matrix, its determinant is `-ε² = -2^-46`, so it has rank two. This occurs with only one token column, without any addition-order change. Consequently, the real bound `rank(AB^T)≤r` is not automatically a rank bound on the reference's rounded gradient. The same warning applies to rounded moment updates.

This example also shows why full exact rank is **not** a universal no-compression proof: the rounded full-rank matrix is still generated by four FP32 factor values plus the known rounding rule. Algebraic low-rank storage and compact computational descriptions are different representation classes. Numerical rank can also be one for this nearly rank-one example depending on tolerance, while exact rank is two.

### B. Sequential moment rounding is not one final rounding

Use the actual stored baseline constants, zero initial state, and scalar gradients `1,2,3,4`. All gradient values and their squares are exactly representable, so there is no gradient-factor or square-rounding ambiguity here. At the fourth update:

| Quantity | Original sequential FP32 | Exact real historical sum, rounded once |
|---|---|---|
| first moment | `0x3f67a786` | `0x3f67a787` |
| second moment | `0x3cf598a2` | `0x3cf598a3` |

Both differ by one binary32 word. The consolidated sums use exact rational powers of the *stored* beta and injection constants, so this discrepancy cannot be blamed on decimal beta substitution or inaccurate summation. It isolates the missing intermediate roundings.

Doing the consolidation in FP64, using an exact accumulator, or obtaining mathematically exact low-rank factors does not generally restore the reference's intermediate-rounding history. They can improve agreement with an ideal real recurrence while moving away from the specified FP32 trajectory. Conversely, accidental equality on particular inputs is possible; no claim is made that every consolidation always differs.

## 6. Exact correction perspective and audit requirements

One may define exact real residuals for the reference state,

`E_m,t = m_t - (b1 m_(t-1) + c1 g_t)`,

`E_v,t = v_t - (b2 v_(t-1) + c2 (g_t⊙g_t))`.

Then the unrolled exact-real formula includes the weighted sum of these residual matrices. For `v`, this residual definition includes intermediate square rounding as well as the multiplication/addition rounding. Likewise write `g_t=G_t+Δ_t` to expose factor-reconstruction rounding. These identities locate the information omitted by naive consolidation; they do not prove that the residuals are incompressible or affordable. Exact residuals may require more bits than one FP32 word. A residual codec must measure total bytes, reconstruction equality, and temporary peak memory.

For the main experiment:

1. Keep gradient reconstruction fidelity separate from moment consolidation fidelity. Compare against the original per-step words, not only a real formula.
2. Label exact-rational identity checks, floating-point error comparisons, SVD numerical rank, and exact rank as different evidence. Rank upper bounds should never be printed as measured ranks.
3. If ranking stored FP32 matrices, interpret their finite entries exactly for any exact-rank certificate. Rational elimination or a certified nonzero minor can prove rank; a large singular value count at a chosen tolerance is only numerical evidence.
4. Test frozen captured factors/gradients first. A changed optimizer state changes later weights and generally later gradients; matching a frozen-history diagnostic does not itself establish a live training trajectory.
5. Report failures and dense fallbacks. Do not replace a failed bitwise target with an undeclared tolerance, changed beta, reset state, diagonal-only second moment, dropped history, or a different optimizer.
6. A bounded experiment can disprove a proposed universal identity with one counterexample, but cannot establish that all possible exact Adam representations need dense state.

## Reproducible independent checks

Run `python3 independent_algebra_checks.py` in this directory. It uses only Python's standard library and tiny matrices. Binary32 rounding is implemented with exact integer quotient/remainder and ties-to-even logic, with normal/subnormal/tie checks. Exact real identities use `fractions.Fraction`, not an FP64 proxy. The result is `independent_algebra_checks.json`.

Checks passed:

- Three-step, `3×4`, two-token-per-step first-moment concatenation identity.
- Per-step Hadamard-square symmetric lift and historical second-moment identity.
- Separate stored-constant versus `1-beta` distinctions.
- Exact-rank increase in a rounded rank-one `2×2` outer product.
- Four-step scalar counterexample to once-rounded first and second moments.
- Explicit intra-step/cross-step cross-term controls.

No literature claim is needed for these finite algebraic identities and directly executable counterexamples. The local baseline source and the self-contained checker are the primary evidence.

## Independent review of the completed main grid

The main `consolidation.py` grid subsequently became available. `independent_grid_review.py` reads its raw archived matrices and codec files without importing the main implementation or using NumPy. It uses this audit's separate integer RN32 implementation, exact rational recurrences, a separate forward Gaussian elimination, and modular elimination over prime `2147483647`.

All 36 reported cases passed:

- 10,368 gradient words reconstructed independently from the recorded factors.
- 20,736 sequential moment words reproduced independently.
- 10,368 once-rounded consolidated moment words reproduced independently.
- 36 rational factor files independently decoded, consumed exactly, and reconstructed to the full exact rational moment matrices; all recorded byte counts matched.
- 216 reported rank values independently matched by exact rational elimination. The modular rank lower bound also matched all 216 values. This is not tolerance-based numerical-rank evidence.
- Reported first/second-moment mismatch counts independently matched the raw words.

The reviewed main source SHA-256 was `04240501d182f0f1490a15fb4f3afe53d79aa5833e8f629ec3ea50b054735cdc`. Detailed machine-readable evidence is in `independent_grid_review.json`. The original grid's 27/36 mismatching cases in each of `m`, `v`, and the diagnostic update are consistent with the audit: all nine one-step cases match, while all 27 reported multi-step cases differ. Its update comparison holds current weights common and does not claim a coupled training run.

No arithmetic or codec defect was found in the reviewed implementation. The reported rational-file size is a serialization result, not a runtime peak-memory bound or a lower bound on every possible codec. Its FP32-factor-payload column is explicitly hypothetical; arbitrary rational factor values cannot be assumed to fit in FP32 merely because the rank is small.

## Focused audit of the fixed-power-of-two positive control

`controls.py` has a legitimate restricted exception: starting with zero moments, every supplied gradient is `g_t,ij = c_t 2^(p_i+q_j)` for the same fixed integer row/column exponents. Binary32 rounding commutes with the power-of-two rescaling while all relevant nonzero intermediate quantities stay within the normal finite range. Therefore one sequential FP32 scalar first moment and one scalar second moment reconstruct the full matrix moments using exponents `p_i+q_j` and `2(p_i+q_j)` respectively. This preserves the original stepwise roundings rather than replacing them with one final rounded historical sum.

`independent_fixed_power2_review.py` independently reproduced all 8,192 moment words across 64 archived scalar inputs, checked all 128 archived scalar-state words and 128 final matrix words, and verified that all nonzero input/intermediate/output values in this replay were normal and finite. It did not rerun the main grid. Details are in `independent_fixed_power2_review.json`.

The proposed packed persistent payload is correctly counted as `8 + 16 + 16 + 16 = 56 B`: two moments, row/column exponent pattern, four-word shape/step/format metadata, and four coefficients. This is **10.9375% of 512 B**, or an 89.0625% reduction, not a below-ten-percent result. As reviewed, this is a representation-size calculation, not a measured 56-byte Python runtime or a demonstrated packed-format decoder. The source already excludes reconstruction workspace, temporaries, stack, weights, and forward/backward training. A finite per-step input scalar is supplied externally just as a normal optimizer is supplied its current gradient; generating or retaining a whole input stream is outside this persistent moment payload.

Further limits to retain explicitly: generic nonzero initial moments are not represented; the same positive power-of-two pattern must persist at every update; the safe-range condition applies to intermediate products/squares/additions as well as final states; and only this 64-step moment test was executed. Adam's elementwise square root, epsilon, parameter update, and a coupled model do not guarantee that future training gradients retain this pattern. The positive control establishes that special exact compact states exist, not that general Adam histories can be merged into this form.
