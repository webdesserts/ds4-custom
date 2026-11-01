/* The Rust DSpark draft-length rule (adapter/dspark-policy) against the
 * original C rule it replaced: identical decisions at every row, for confidence
 * sequences that cover every decile edge, and for the values tested by
 * DS4_DSPARK_GAIN. Built with ds4.c's flags (fast-math included). */
#include "adapter/include/dspark_policy.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The original C rule (9f1f59f), frozen here as the reference the Rust rule
 * must reproduce. Updates `*survive` with this row's acceptance estimate. */
static inline bool ds4_dspark_gain_continue_c(float value, uint32_t row, float confidence,
                                              float *survive) {
    static const float accept[10] = {
        0.0500f, 0.0605f, 0.3206f, 0.3053f, 0.4847f,
        0.5770f, 0.6446f, 0.7688f, 0.9096f, 0.9937f,
    };
    /* ms added by verifying draft row+1 instead of row drafts (row 0: the
     * 2-row verify instead of a single-token decode). */
    static const float marginal_ms[DS4_DSPARK_GAIN_ROWS] = {8.28f, 6.84f, 7.18f, 8.83f, 10.48f};
    int bucket = (int)(confidence * 10.0f);
    if (!(bucket >= 0)) bucket = 0;
    if (bucket > 9) bucket = 9;
    *survive *= accept[bucket];
    return *survive > value * marginal_ms[row < DS4_DSPARK_GAIN_ROWS ? row : DS4_DSPARK_GAIN_ROWS - 1];
}

static const float values_tok_s[] = {1, 10, 30, 50, 60, 65, 70, 80, 90, 100, 120, 150, 400};

/* Compares every prefix of `conf`; returns 0 on a mismatch. */
static int same_decisions(float value_tok_s, const float *conf, uint32_t n, long *checked) {
    const ds4_dspark_gain_model model = ds4_dspark_gain_model_default(value_tok_s);
    const float c_value = value_tok_s / 1000.0f;
    float survive = 1.0f;
    for (uint32_t row = 0; row < n; row++) {
        const bool c = ds4_dspark_gain_continue_c(c_value, row, conf[row], &survive);
        const bool rust = ds4_dspark_gain_continue(&model, conf, row + 1);
        ++*checked;
        if (c != rust) {
            fprintf(stderr, "mismatch value=%g row=%u conf[row]=%.9g c=%d rust=%d\n",
                    value_tok_s, row, conf[row], c, rust);
            return 0;
        }
        if (!c) break; /* the server stops drafting here */
    }
    return 1;
}

int main(void) {
    /* Decile edges and their neighbours, mid-decile values and the clamps. */
    float grid[64];
    uint32_t n_grid = 0;
    for (int k = 0; k <= 10; k++) {
        const float edge = k / 10.0f;
        grid[n_grid++] = edge;
        grid[n_grid++] = nextafterf(edge, -1.0f);
        grid[n_grid++] = nextafterf(edge, 2.0f);
        if (k < 10) grid[n_grid++] = edge + 0.05f;
    }
    grid[n_grid++] = -1.0f;
    grid[n_grid++] = 3.0f;
    long checked = 0;
    for (size_t v = 0; v < sizeof(values_tok_s) / sizeof(*values_tok_s); v++) {
        float conf[5];
        for (uint32_t a = 0; a < n_grid; a++)
            for (uint32_t b = 0; b < n_grid; b++)
                for (uint32_t c = 0; c < n_grid; c++) {
                    conf[0] = grid[a]; conf[1] = grid[b]; conf[2] = grid[c];
                    if (!same_decisions(values_tok_s[v], conf, 3, &checked)) return 1;
                }
        uint32_t seed = 12345u + (uint32_t)v;
        for (int i = 0; i < 200000; i++) {
            for (int r = 0; r < 5; r++) {
                seed = seed * 1664525u + 1013904223u;
                conf[r] = r < 2 && (seed >> 31) ? 0.9f + (seed >> 8) % 1000 / 10000.0f
                                                : (seed >> 8) / 16777216.0f;
            }
            if (!same_decisions(values_tok_s[v], conf, 5, &checked)) return 1;
        }
    }
    /* Boundary values: for every decile sequence of up to five rows, value the
     * last row exactly at the C rule's break-even point and one ulp either
     * side, so the comparison, the table and the product order all matter. */
    const float mid[10] = {.05f, .15f, .25f, .35f, .45f, .55f, .65f, .75f, .85f, .95f};
    const float marginal[DS4_DSPARK_GAIN_ROWS] = {8.28f, 6.84f, 7.18f, 8.83f, 10.48f};
    long boundary = 0;
    for (uint32_t n = 1; n <= 5; n++) {
        uint32_t total = 1;
        for (uint32_t r = 0; r < n; r++) total *= 10;
        for (uint32_t code = 0; code < total; code++) {
            float conf[5], survive = 1.0f;
            for (uint32_t r = 0, x = code; r < n; r++, x /= 10) conf[r] = mid[x % 10];
            for (uint32_t r = 0; r + 1 < n; r++) (void)ds4_dspark_gain_continue_c(0.0f, r, conf[r], &survive);
            float probe = survive;
            (void)ds4_dspark_gain_continue_c(0.0f, n - 1, conf[n - 1], &probe);
            const float even = probe / marginal[n - 1];
            const float values[3] = {even, nextafterf(even, 0.0f), nextafterf(even, 1.0f)};
            for (int k = 0; k < 3; k++) {
                ds4_dspark_gain_model model = ds4_dspark_gain_model_default(0.0f);
                model.value = values[k];
                float s = survive;
                const bool c = ds4_dspark_gain_continue_c(values[k], n - 1, conf[n - 1], &s);
                if (c != ds4_dspark_gain_continue(&model, conf, n)) {
                    fprintf(stderr, "boundary mismatch n=%u code=%u value=%.9g c=%d\n", n, code, values[k], c);
                    return 1;
                }
                boundary++;
            }
        }
    }
    checked += boundary;
    /* A null model and an unset value mean the default model. */
    const float one[1] = {0.55f};
    const ds4_dspark_gain_model def = ds4_dspark_gain_model_default(0.0f);
    if (ds4_dspark_gain_continue(NULL, one, 1) != ds4_dspark_gain_continue(&def, one, 1)) return 1;
    printf("dspark policy: Rust rule equals the C rule at %ld row decisions: PASS\n", checked);
    return 0;
}
