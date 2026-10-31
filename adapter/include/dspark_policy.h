/* V4.1 DSpark draft length (adapter/dspark-policy/src/lib.rs): keep drafting
 * while P(every draft so far is accepted) exceeds the next row's marginal
 * verify cost valued at `value` tokens/ms. A pure function of the rows'
 * sigmoid confidences and the model; dspark-policy-sim calls the same code. */
#ifndef DS4_DSPARK_POLICY_H
#define DS4_DSPARK_POLICY_H

#include <stdbool.h>
#include <stdint.h>

enum { DS4_DSPARK_GAIN_BUCKETS = 10, DS4_DSPARK_GAIN_ROWS = 5 };

typedef struct {
    float accept[DS4_DSPARK_GAIN_BUCKETS];  /* P(accept | earlier accepted) by confidence decile */
    float marginal_ms[DS4_DSPARK_GAIN_ROWS]; /* ms added by verifying draft row i */
    float value;                             /* tokens per ms */
} ds4_dspark_gain_model;

/* Whether to keep drafting after `n` rows (newest last). NULL model: default. */
bool ds4_dspark_gain_continue(const ds4_dspark_gain_model *model, const float *confidences, uint32_t n);

/* The fitted default tables valued at `value_tok_s` tokens per second. */
ds4_dspark_gain_model ds4_dspark_gain_model_default(float value_tok_s);

/* The original C rule (9f1f59f), kept for DS4_DSPARK_POLICY_C=1 and as the
 * reference tests/test_dspark_policy.c compares the Rust rule with. Updates
 * `*survive` with this row's acceptance estimate. */
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

#endif
