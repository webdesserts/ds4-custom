/* ds4-side smoke test of the pinned cache-reuse-utils and kv-archive-store C
 * ABIs as the native adapter uses them. Host-only: links just the static
 * planner archive. */
#include "cache_reuse_utils.h"
#include "kv_archive_store.h"
#include <stdio.h>
#include <string.h>

#define EXPECT(cond, msg) do { if (!(cond)) { fprintf(stderr, "%s\n", msg); return 1; } } while (0)

/* The coordinate, range and selection helpers the SCR engine delegates to. */
static int smoke_engine_helpers(void) {
    const int32_t physical[5] = {5, 0, 6, 0, 7}, logical[3] = {5, 6, 7};
    const uint32_t pads[2] = {1, 3};
    bool ok = false;
    EXPECT(cache_reuse_verify_matches_logical(physical, 5, pads, 2, logical, 3, &ok, NULL, 0) == 0 && ok,
           "verify_matches_logical refused a valid map");
    EXPECT(cache_reuse_verify_matches_logical(physical, 5, pads, 1, logical, 3, &ok, NULL, 0) == 0 && !ok,
           "verify_matches_logical accepted a missing pad");
    uint32_t out[4] = {0}, len = 0;
    EXPECT(cache_reuse_slice_pad_positions(pads, 2, 5, 2, 3, out, 4, &len, NULL, 0) == 0 &&
           len == 1 && out[0] == 1, "slice_pad_positions");
    EXPECT(cache_reuse_truncate_pad_positions(pads, 2, 5, 3, &len, NULL, 0) == 0 && len == 1,
           "truncate_pad_positions");
    EXPECT(cache_reuse_append_unpadded(5, pads, 2, 2, &len, NULL, 0) == 0 && len == 7,
           "append_unpadded");
    EXPECT(cache_reuse_append_unpadded(UINT32_MAX, NULL, 0, 1, &len, NULL, 0) == CACHE_REUSE_ERR_INVALID,
           "append_unpadded overflow was not rejected");
    const uint32_t dst_pads[1] = {12};
    cache_reuse_range span = {2, 0, 4, 10};
    EXPECT(cache_reuse_verify_span_correspondence((const uint32_t[]){4}, 1, 8, dst_pads, 1, 16,
               &span, &ok, NULL, 0) == 0 && ok, "verify_span_correspondence");
    cache_reuse_range planned[2] = {{4, 4, 8, 4}, {16, 16, 8, 16}}, clipped[2];
    EXPECT(cache_reuse_clip_ranges_to_archive(planned, 2, 0, 100, 19, 2, clipped, 2, &len, NULL, 0) == 0 &&
           len == 2 && clipped[1].count == 2, "clip_ranges_to_archive");
    cache_reuse_range executed[2] = {{4, 4, 4, 4}, {16, 16, 8, 16}};
    cache_reuse_execution_summary summary;
    EXPECT(cache_reuse_reconcile_executed_ranges(planned, 2, executed, 2, 32, 2, &summary, NULL, 0) == 0 &&
           summary.reused_tokens == 12 && summary.gap_tokens == 8 && summary.suffix_tokens == 8,
           "reconcile_executed_ranges");
    const int32_t target[3] = {1, 2, 3}, hot_tokens[2] = {1, 9};
    cache_reuse_candidate hot = {hot_tokens, 2, NULL, 0, 0}, pool = {target, 3, NULL, 0, 0};
    cache_reuse_candidate_policy policy = {1, 1, 1};
    cache_reuse_candidate_selection pick;
    EXPECT(cache_reuse_select_candidate(target, 3, &hot, &pool, 1, &policy, &pick, NULL, 0) == 0 &&
           pick.action == CACHE_REUSE_ACTION_RESTORE_POOL && pick.pool_index == 0 &&
           pick.reusable_tokens == 3, "select_candidate");
    /* Two records of two layers, two bytes per row, gathered layer-major. */
    const uint8_t window[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    uint8_t gathered[8] = {0};
    const uint8_t layer_major[8] = {0, 1, 4, 5, 2, 3, 6, 7};
    EXPECT(kv_archive_gather_layers(window, 8, 2, 2, 2, gathered, 8, 2, NULL, 0) == KV_ARCHIVE_OK &&
           !memcmp(gathered, layer_major, 8), "kv_archive_gather_layers");
    EXPECT(kv_archive_gather_layers(window, 8, 2, 2, 2, gathered, 4, 1, NULL, 0) == KV_ARCHIVE_ERR_INVALID,
           "short kv gather output was not rejected");
    return 0;
}

static cache_reuse_config config_for(uint32_t k) {
    cache_reuse_config c;
    memset(&c, 0, sizeof(c));
    c.max_ranges = k;
    c.minimum_match_tokens = 1;
    c.minimum_physical_reuse_tokens = 1;
    c.source_start_alignment = 1;
    c.prepared_target_start_alignment = 1;
    c.reuse_count_alignment = 1;
    c.minimum_prepared_reuse_start = 0;
    c.protected_prefix_tokens = 0;
    c.fresh_suffix_tokens = 0;
    c.max_trim_left = 0;
    c.max_trim_right = 0;
    c.pad_token = -1;
    c.loss_basis_points = 1000;
    c.max_new_padding = 0;
    c.max_prepared_tokens = 20;
    c.variant_limit = 4096;
    c.budget_ms = 5000;
    c.allow_all_insertions = false;
    c.allowed_insertions = NULL;
    c.allowed_insertion_count = 0;
    /* One-token toy matches: a nominal per-range charge keeps the multi-range
     * shape observable; 0 (the ds4 adapter setting) selects the tuned 16/32. */
    c.pad_penalty_tokens = 1;
    c.range_penalty_tokens = 1;
    return c;
}

static int plan(const int32_t *target, uint32_t tn, const int32_t *source, uint32_t sn,
                cache_reuse_config *cfg, cache_reuse_result *r, bool (*cancel)(void *)) {
    char err[256] = {0};
    int rc = cache_reuse_plan(target, tn, source, sn, 7, cfg, r, err, sizeof(err), cancel, NULL);
    if (rc && err[0]) fprintf(stderr, "planner error: %s\n", err);
    return rc;
}

static bool always_cancel(void *ud) { (void)ud; return true; }

_Static_assert(sizeof(cache_reuse_config) == 96, "pinned config ABI size");
_Static_assert(sizeof(cache_reuse_result) == 1104, "pinned result ABI size");
_Static_assert(CACHE_REUSE_MAX_RANGES == 32 && CACHE_REUSE_MAX_INSERTIONS == 32, "pinned limits");

int main(void) {
    int32_t target[20] = {10,0,11,0,12,0,13,0,14,0,15,0,16,0,17,0,18,0,19,0};
    int32_t source[20] = {10,99,11,99,12,99,13,99,14,99,15,99,16,99,17,99,18,99,19,99};
    cache_reuse_config cfg = config_for(10);
    cache_reuse_result r;
    memset(&r, 0, sizeof(r));
    if (plan(target, 20, source, 20, &cfg, &r, NULL) != 0 ||
        r.status != CACHE_REUSE_COMPLETE || r.range_count != 10 || r.covered_tokens != 10) {
        fprintf(stderr, "ten-range plan failed status=%u ranges=%u covered=%u\n",
                r.status, r.range_count, r.covered_tokens);
        return 1;
    }
    cfg = config_for(10);
    cfg.variant_limit = 1;
    memset(&r, 0, sizeof(r));
    if (plan(target, 20, source, 20, &cfg, &r, NULL) != 0 ||
        r.status != CACHE_REUSE_VARIANT_LIMIT || r.range_count == 0) {
        fprintf(stderr, "limited valid incumbent was not preserved: status=%u ranges=%u\n",
                r.status, r.range_count);
        return 1;
    }
    cfg = config_for(CACHE_REUSE_MAX_RANGES);
    memset(&r, 0, sizeof(r));
    if (plan(target, 20, source, 20, &cfg, &r, NULL) != 0 ||
        r.status != CACHE_REUSE_COMPLETE || r.range_count != 10) {
        fprintf(stderr, "K32 plan failed status=%u ranges=%u\n", r.status, r.range_count);
        return 1;
    }
    cfg = config_for(CACHE_REUSE_MAX_RANGES + 1u);
    memset(&r, 0, sizeof(r));
    if (plan(target, 20, source, 20, &cfg, &r, NULL) == 0 || r.status != CACHE_REUSE_ERR_INVALID) {
        fprintf(stderr, "K33 was not rejected\n");
        return 1;
    }
    /* No qualifying match at all is a complete empty plan (fresh prefill),
     * not NO_FEASIBLE; NO_FEASIBLE is reserved for unplaceable matches. */
    int32_t unrelated[20] = {50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69};
    cfg = config_for(CACHE_REUSE_MAX_RANGES);
    memset(&r, 0, sizeof(r));
    if (plan(unrelated, 20, source, 20, &cfg, &r, NULL) != 0 ||
        r.status != CACHE_REUSE_COMPLETE || r.range_count != 0) {
        fprintf(stderr, "no-match plan was not complete/empty: status=%u ranges=%u\n",
                r.status, r.range_count);
        return 1;
    }
    /* Historical source pad token IDs are ordinary content at this boundary:
     * the core receives actual tokens only and can still match an equal target. */
    int32_t padded_target[7] = {1, 77, 2, 3, 88, 4, 5};
    int32_t padded_source[7] = {1, 77, 2, 3, 88, 4, 5};
    cfg = config_for(10);
    cfg.max_prepared_tokens = 7;
    memset(&r, 0, sizeof(r));
    if (plan(padded_target, 7, padded_source, 7, &cfg, &r, NULL) != 0 ||
        r.range_count != 1 || r.covered_tokens != 7) {
        fprintf(stderr, "historical source token content test failed\n");
        return 1;
    }
    int32_t insert_target[6] = {1, 2, 3, 4, 5, 9};
    int32_t insert_source[4] = {2, 3, 4, 5};
    uint32_t allowed_boundary[1] = {0};
    cfg = config_for(1);
    cfg.minimum_match_tokens = 4;
    cfg.max_prepared_tokens = 7;
    cfg.max_new_padding = 1;
    cfg.pad_token = 77;
    cfg.prepared_target_start_alignment = 2;
    cfg.allowed_insertions = allowed_boundary;
    cfg.allowed_insertion_count = 1;
    memset(&r, 0, sizeof(r));
    if (plan(insert_target, 6, insert_source, 4, &cfg, &r, NULL) != 0 ||
        r.range_count != 1 || r.insertion_count != 1 ||
        r.insertions[0].token != 77 || r.insertions[0].count != 1 ||
        r.ranges[0].prepared_start != 2) {
        fprintf(stderr, "authorized insertion recipe test failed\n");
        return 1;
    }
    if (r.discovered_match_count == 0 || r.longest_match_tokens != 4) {
        fprintf(stderr, "discovery stats missing: matches=%u longest=%u\n",
                r.discovered_match_count, r.longest_match_tokens);
        return 1;
    }
    /* Materialize the planned prepared target and walk its coordinates. */
    {
        char err[256] = {0};
        int32_t prepared[7] = {0};
        uint32_t pads[1] = {UINT32_MAX};
        if (r.prepared_tokens != 7 || r.inserted_padding != 1 ||
            cache_reuse_materialize(insert_target, 6, insert_source, 4, &r, prepared, 7,
                                    pads, 1, err, sizeof(err)) != CACHE_REUSE_COMPLETE ||
            pads[0] != r.insertions[0].prepared_start || prepared[pads[0]] != 77) {
            fprintf(stderr, "materialize failed: %s\n", err);
            return 1;
        }
        int32_t logical[7] = {0};
        uint32_t logical_len = 99;
        if (cache_reuse_project_logical(prepared, 7, pads, 1, logical, 7, &logical_len,
                                        err, sizeof(err)) != CACHE_REUSE_COMPLETE ||
            logical_len != 6 || memcmp(logical, insert_target, sizeof(insert_target))) {
            fprintf(stderr, "logical projection failed: %s\n", err);
            return 1;
        }
        uint32_t boundary = 99;
        if (cache_reuse_logical_boundary(pads[0] + 1u, 7, pads, 1, &boundary,
                                         err, sizeof(err)) != CACHE_REUSE_COMPLETE ||
            boundary != pads[0]) {
            fprintf(stderr, "logical boundary failed: %s\n", err);
            return 1;
        }
        if (cache_reuse_logical_boundary(8, 7, pads, 1, &boundary, NULL, 0) !=
                CACHE_REUSE_ERR_INVALID || boundary != 0) {
            fprintf(stderr, "out-of-range boundary was not rejected\n");
            return 1;
        }
        /* The next request keeps the logical prefix 1,2,3 and diverges. */
        int32_t next[5] = {1, 2, 3, 40, 41};
        cache_reuse_common_prefix_result common;
        memset(&common, 0xff, sizeof(common));
        if (cache_reuse_common_prefix(next, 5, prepared, 7, pads, 1, &common,
                                      err, sizeof(err)) != CACHE_REUSE_COMPLETE ||
            common.common_logical_tokens != 3 || common.carried_pads != 1 ||
            common.physical_prefix_tokens != 4 || common.live_logical_tokens != 6) {
            fprintf(stderr, "common prefix failed: %s\n", err);
            return 1;
        }
        uint32_t unordered[2] = {3, 1};
        if (cache_reuse_common_prefix(next, 5, prepared, 7, unordered, 2, &common,
                                      NULL, 0) != CACHE_REUSE_ERR_INVALID ||
            common.common_logical_tokens != 0) {
            fprintf(stderr, "unordered pad positions were not rejected\n");
            return 1;
        }
    }
    cfg = config_for(1);
    cfg.minimum_match_tokens = 4;
    cfg.max_prepared_tokens = 7;
    cfg.max_new_padding = 1;
    cfg.pad_token = 77;
    cfg.prepared_target_start_alignment = 2;
    cfg.allowed_insertions = NULL;
    cfg.allowed_insertion_count = 0;
    memset(&r, 0, sizeof(r));
    /* Without insertion authority the planner left-trims to the alignment. */
    if (plan(insert_target, 6, insert_source, 4, &cfg, &r, NULL) != 0 ||
        r.status != CACHE_REUSE_COMPLETE || r.insertion_count != 0 ||
        r.range_count != 1 || r.ranges[0].prepared_start % 2u) {
        fprintf(stderr, "disallowed insertion was not trimmed: status=%u ranges=%u insertions=%u\n",
                r.status, r.range_count, r.insertion_count);
        return 1;
    }
    /* A discovered match that no placement may use is NO_FEASIBLE. */
    cfg = config_for(1);
    cfg.minimum_match_tokens = 4;
    cfg.max_prepared_tokens = 7;
    cfg.minimum_prepared_reuse_start = 6;
    memset(&r, 0, sizeof(r));
    if (plan(insert_target, 6, insert_source, 4, &cfg, &r, NULL) != 0 ||
        r.status != CACHE_REUSE_NO_FEASIBLE || r.range_count != 0) {
        fprintf(stderr, "unplaceable match was not NO_FEASIBLE: status=%u ranges=%u\n",
                r.status, r.range_count);
        return 1;
    }
    cfg = config_for(10);
    memset(&r, 0, sizeof(r));
    if (plan(NULL, 20, source, 20, &cfg, &r, NULL) == 0 || r.status != CACHE_REUSE_ERR_INVALID) {
        fprintf(stderr, "null target was not rejected\n");
        return 1;
    }
    if (r.timing_known || r.elapsed_ns != 0) {
        fprintf(stderr, "unmeasured invalid-input timing was reported as known\n");
        return 1;
    }
    int32_t negative[20] = {10,0,11,0,12,0,13,0,14,0,15,0,16,0,17,0,18,0,19,0};
    negative[3] = -1;
    memset(&r, 0, sizeof(r));
    if (plan(target, 20, negative, 20, &cfg, &r, NULL) == 0 || r.status != CACHE_REUSE_ERR_INVALID) {
        fprintf(stderr, "negative token was not rejected\n");
        return 1;
    }
    cfg = config_for(10);
    cfg.budget_ms = 0;
    memset(&r, 0, sizeof(r));
    if (plan(target, 20, source, 20, &cfg, &r, NULL) == 0 || r.status != CACHE_REUSE_ERR_BUDGET) {
        fprintf(stderr, "zero budget was not propagated\n");
        return 1;
    }
    if (!r.timing_known || r.elapsed_ns == 0 || r.preparation_ns == 0 ||
        r.discovery_entered || r.search_entered) {
        fprintf(stderr, "zero-budget timing was not preserved as measured preparation\n");
        return 1;
    }
    cfg = config_for(10);
    memset(&r, 0, sizeof(r));
    if (plan(target, 20, source, 20, &cfg, &r, always_cancel) == 0 ||
        r.status != CACHE_REUSE_ERR_CANCELLED) {
        fprintf(stderr, "cancel was not propagated\n");
        return 1;
    }
    if (smoke_engine_helpers()) return 1;
    puts("cache_reuse_ffi_test: ok");
    return 0;
}
