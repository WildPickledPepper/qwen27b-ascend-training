/* Fused BF16 gradient accumulation kernels (kernel only, no torch dependency). */
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static inline float bf2f(uint16_t v) {
    uint32_t u = (uint32_t)v << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static inline uint16_t f2bf(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;          /* round to nearest even, same as torch bf16 add */
    return (uint16_t)(u >> 16);
}

/* acc[i] += g[i] : identical semantics and bitwise result to torch bf16 add_ */
void accum1(uint16_t *acc, const uint16_t *g, size_t n) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++)
        acc[i] = f2bf(bf2f(acc[i]) + bf2f(g[i]));
}

/* acc[i] += sum_j gs[j][i] : accumulator touched once per k microbatches.
 * Intentionally NOT bitwise equal to k sequential accumulations. */
void accumk(uint16_t *acc, uint16_t *const *gs, int k, size_t n) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) {
        float s = bf2f(gs[0][i]);
        for (int j = 1; j < k; j++) s += bf2f(gs[j][i]);
        acc[i] = f2bf(bf2f(acc[i]) + s);
    }
}
