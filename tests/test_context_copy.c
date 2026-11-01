#include "../ds4_context_copy.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#define N(a) ((uint32_t)(sizeof(a) / sizeof(*(a))))

static void expect(const int *ctx, uint32_t n, int next, uint32_t min_match,
                   uint32_t depth, const int *want) {
    int out[8];
    memset(out, 0x7f, sizeof(out));
    const bool found = ds4_context_copy_find(ctx, n, next, min_match, depth, out);
    if (!want) {
        assert(!found);
        return;
    }
    assert(found);
    assert(!memcmp(out, want, depth * sizeof(*out)));
}

int main(void) {
    /* The span after the earlier "1 2 3". */
    const int basic[] = {1, 2, 3, 4, 5, 6, 7, 1, 2};
    expect(basic, N(basic), 3, 3, 3, (const int[]){4, 5, 6});
    expect(basic, N(basic), 3, 3, 4, (const int[]){4, 5, 6, 7});

    /* Only the key's last token matches: no copy. */
    const int tail_only[] = {9, 9, 3, 4, 5, 6, 1, 2};
    expect(tail_only, N(tail_only), 3, 2, 2, NULL);

    /* The most recent occurrence wins. */
    const int twice[] = {1, 2, 10, 11, 0, 1, 2, 20, 21, 0, 1};
    expect(twice, N(twice), 2, 2, 2, (const int[]){20, 21});

    /* A copy may run into the current occurrence, ending at `next`. Read past
     * the context through a guard so an out-of-bounds read would differ. */
    int guarded[6] = {7, 8, 9, 7, 8, -5};
    expect(guarded, 5, 9, 2, 3, (const int[]){7, 8, 9});

    /* A match too recent to fill the depth is skipped for an older one... */
    const int recent[] = {5, 6, 1, 2, 5, 6, 3, 5};
    expect(recent, N(recent), 6, 2, 3, (const int[]){3, 5, 6});
    expect(recent, N(recent), 6, 2, 4, (const int[]){1, 2, 5, 6});
    /* ...and with no older one there is no copy. */
    expect(recent, N(recent), 6, 2, 8, NULL);

    /* A key at the very start of the context counts. */
    const int start[] = {4, 5, 6, 7, 8, 4};
    expect(start, N(start), 5, 2, 3, (const int[]){6, 7, 8});

    /* Degenerate requests. */
    expect(start, N(start), 5, 0, 3, NULL);
    expect(start, N(start), 5, 2, 0, NULL);
    expect(start, 1, 5, 3, 1, NULL);
    expect(start, 2, 5, 2, 3, NULL);
    expect(start, 2, 5, 2, 4, NULL);

    puts("test_context_copy: ok");
    return 0;
}
