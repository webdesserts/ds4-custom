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

#endif
