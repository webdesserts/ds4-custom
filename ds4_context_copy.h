/* Context-copy drafts: when a reply repeats a span of its context (a quoted
 * file, an edit that rewrites code), the tokens that followed the span's
 * earlier occurrence are a free draft. The target verifies every draft, so a
 * copy changes only speed, never the greedy output. */
#ifndef DS4_CONTEXT_COPY_H
#define DS4_CONTEXT_COPY_H

#include <stdbool.h>
#include <stdint.h>

/* The sequence is ctx[0..n) followed by `next`. Finds the most recent earlier
 * occurrence of its last `min_match` tokens whose following `depth` tokens
 * are all known (they may run into the current occurrence, as in a repeated
 * run), and copies those tokens to out. Returns false, leaving out
 * unspecified, when there is no such occurrence. */
static inline bool ds4_context_copy_find(const int *ctx, uint32_t n, int next,
                                         uint32_t min_match, uint32_t depth, int *out) {
    if (!ctx || !out || !min_match || !depth || min_match > n || depth > n) return false;
    /* s[i] for i <= n; s[n] is `next`. The key ends at s[n]. */
#define DS4_CONTEXT_COPY_AT(i) ((i) == n ? next : ctx[i])
    /* An occurrence ending at j needs j - min_match + 1 >= 0 and j + depth <= n. */
    for (uint32_t j = n - depth + 1u; j-- > min_match - 1u;) {
        if (ctx[j] != next) continue;
        uint32_t k = 1;
        while (k < min_match && ctx[j - k] == ctx[n - k]) k++;
        if (k < min_match) continue;
        for (uint32_t i = 0; i < depth; i++) out[i] = DS4_CONTEXT_COPY_AT(j + 1u + i);
        return true;
    }
#undef DS4_CONTEXT_COPY_AT
    return false;
}

#endif
