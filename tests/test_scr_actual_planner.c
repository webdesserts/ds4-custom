/* End-to-end logical policy through the pinned cache-reuse-utils planner:
 * archive capture/seal, planning, target materialization, install/rebase and
 * publication. Strict host: tests/scr_host_storage.c replaces every GPU buffer
 * and model rows/heads are simulated. Not inference or quality evidence. */
#define DS41_SCR_PLANNER_TEST_HOOKS
#include "scr_engine_harness.h"
static void actual_no_padding_test(ds4_session *s, int *source, int *v, uint32_t target_len, uint32_t block_stride, uint32_t matches, uint32_t target_matches, uint32_t expected_matches, uint32_t expected_status) {
    ds41_gpu_graph *g = &s->ds41_graph;
    ds4_engine *e = s->engine;
    e->vocab.n_vocab = DS4_N_VOCAB;
    e->vocab.token = calloc(DS4_N_VOCAB, sizeof(*e->vocab.token)); CHECK(e->vocab.token);
    for (uint32_t i = 0; i < DS4_N_VOCAB; ++i)
        e->vocab.token[i] = (ds4_str){.ptr = "\xc4\x8a", .len = 2};
    const uint32_t archive_count = 2048u;
    ds4_session_invalidate(s);
    s->checkpoint.len = 0; s->checkpoint_valid = false;
    s->scr_planner_budget_ms = 5000u;
    s->scr_capture_batch_chunks = true;
    for (uint32_t i = 0; i < 4096u; ++i) source[i] = 1000 + (int)i;
    controls(source);
    CHECK(!ds4_session_scr_archive_prepare(s, 8u, archive_count));
    ds41_scr_archive *a = g->archive; CHECK(a);
    for (uint32_t from = 8u; from < 8u + archive_count; from += 128u) {
        capture(g, from, 128u);
        memcpy(s->checkpoint.v, source, (size_t)(from + 128u) * sizeof(int));
        s->checkpoint.len = (int)(from + 128u); g->pos = from + 128u;
        s->checkpoint_valid = g->valid = true;
        CHECK(!ds4_session_scr_archive_flush(s));
    }
    CHECK(!ds4_session_scr_archive_seal(s, source + 8u, archive_count));
    CHECK(a->end == 8u + archive_count);
    const uint64_t source_generation = a->source_generation;
    /* The archive stores the actual source interval. Build the independent
     * original target from those actual tokens; filler is unique. */
    for (uint32_t i = 0; i < target_len; ++i) v[i] = 90000 + (int)i;
    for (uint32_t i = 0; i < target_matches; ++i)
        memcpy(v + 8u + i * block_stride, a->tokens + i * block_stride,
               128u * sizeof(int));
    ds4_tokens full = {.v = v, .len = (int)target_len, .cap = 4096};
    controls(v);
    ds41_even_receipt r; char err[256] = {0};
    int rc = ds41_even_policy(s, &full, archive_count, matches, 201, &r, err, sizeof(err));
    if (rc) fprintf(stderr, "actual K%u no-pad: %s\n", matches, err);
    CHECK(!rc);
    if (!(r.planner_status_known && r.planner_status == expected_status))
        fprintf(stderr, "actual status mismatch expected=%u got=%u known=%d matches=%u expected_matches=%u\n",
            expected_status, r.planner_status, r.planner_status_known, matches, expected_matches);
    CHECK(r.planner_status_known && r.planner_status == expected_status);
    CHECK(r.planner_timing_known && r.planner_discovery_known && r.planner_search_known);
    CHECK(r.planner_elapsed_ns > 0u && r.planner_preparation_ns > 0u);
    CHECK(r.published && r.execution.route == DS4_SCR_DENSE_REUSE);
    CHECK(r.logical_match.selected_ranges == expected_matches);
    CHECK(r.plan.range_count == expected_matches);
    CHECK(r.execution.multi.normalized_spans == expected_matches);
    CHECK(r.execution.multi.committed_spans == expected_matches);
    CHECK(r.execution.multi.committed_tokens == expected_matches * 128u);
    CHECK(r.new_pads == 0u && r.carried_pads == 0u && r.imported_pads == 0u);
    CHECK(r.physical_tokens == target_len && s->checkpoint.len == (int)target_len);
    CHECK(!memcmp(s->checkpoint.v, v, (size_t)target_len * sizeof(int)));
    CHECK(s->scr_coordinates.count == 0u);
    CHECK(s->scr_coordinates.physical_tokens == target_len);
    /* The rebased source may become a retired predecessor; retain the value
     * before the call and never dereference the local old-archive pointer. */
    CHECK(source_generation != 0u && r.execution.source_generation == source_generation);
    CHECK(!ds4_session_scr_archive_release(s));
    free(e->vocab.token); e->vocab.token = NULL; e->vocab.n_vocab = 0;
    printf("actual K%u no-padding target=%u stride=%u: %u aligned spans, install/rebase, publication passed\n",
        matches, target_len, block_stride, expected_matches);
}

static void actual_padding_boundary_test(ds4_session *s, int *source, int *v) {
    ds41_gpu_graph *g = &s->ds41_graph;
    ds4_engine *e = s->engine;
    const uint32_t target_len = 1336u, stride = 132u, matches = 10u, archive_count = 2048u;
    e->vocab.n_vocab = DS4_N_VOCAB;
    e->vocab.token = calloc(DS4_N_VOCAB, sizeof(*e->vocab.token)); CHECK(e->vocab.token);
    for (uint32_t i = 0; i < DS4_N_VOCAB; ++i)
        e->vocab.token[i] = (ds4_str){.ptr = "\xc4\x8a", .len = 2};
    /* GPT-2 byte-level encoding of the ordinary space is U+0120. */
    e->vocab.token[201] = (ds4_str){.ptr = "\xc4\xa0", .len = 2};
    ds4_session_invalidate(s);
    s->checkpoint.len = 0; s->checkpoint_valid = false;
    s->scr_planner_budget_ms = 5000u;
    s->scr_capture_batch_chunks = true;
    s->scr_padding_policy = DS4_SCR_PADDING_PLANNER_FRESH_GAP_SPACE;
    s->scr_padding_selected = true; s->scr_padding_token = 201;
    for (uint32_t i = 0; i < 4096u; ++i) source[i] = 1000 + (int)i;
    /* A historical source pad is ordinary content here, not a new target pad. */
    source[8u + 5u] = 201;
    controls(source);
    CHECK(!ds4_session_scr_archive_prepare(s, 8u, archive_count));
    ds41_scr_archive *a = g->archive; CHECK(a);
    for (uint32_t from = 8u; from < 8u + archive_count; from += 128u) {
        capture(g, from, 128u);
        memcpy(s->checkpoint.v, source, (size_t)(from + 128u) * sizeof(int));
        s->checkpoint.len = (int)(from + 128u); g->pos = from + 128u;
        s->checkpoint_valid = g->valid = true;
        CHECK(!ds4_session_scr_archive_flush(s));
    }
    CHECK(!ds4_session_scr_archive_seal(s, source + 8u, archive_count));
    CHECK(a->end == 8u + archive_count && a->tokens[5] == 201);
    /* Attach genuine historical source-pad provenance, not merely an equal-ID
     * ordinary token: the archive owns and is charged for this one-record map. */
    size_t source_map_bytes = 0;
    CHECK(ds41_coordinates_heap_bytes(1u, &source_map_bytes));
    const uint32_t source_pad = 5u;
    CHECK(ds41_coordinates_build(&a->coordinates, a->tokens, archive_count,
        DS4_N_VOCAB, &source_pad, 1u, a->source_generation));
    a->allocation_bytes += source_map_bytes;
    g->allocation_bytes += source_map_bytes;
    e->ds41_session_bytes += source_map_bytes;
    const uint64_t source_generation = a->source_generation;
    const uint64_t source_map_identity = a->coordinates.identity;
    CHECK(a->coordinates.count == 1u && a->coordinates.pads[0] == 5u && a->tokens[5] == 201 &&
        source_map_identity == source_generation);
    /* The target is shifted one token right of the source blocks. The native
     * planner must insert one pad at target index 9; the last selected span
     * ends at 1325 while the original suffix is only nine tokens. */
    for (uint32_t i = 0; i < target_len; ++i) v[i] = 90000 + (int)i;
    for (uint32_t i = 0; i < matches; ++i)
        memcpy(v + 9u + i * stride, a->tokens + i * stride, 128u * sizeof(int));
    int expected[4096];
    memcpy(expected, v, 9u * sizeof(int));
    expected[9] = 201;
    memcpy(expected + 10u, v + 9u, (target_len - 9u) * sizeof(int));
    ds4_tokens full = {.v = v, .len = (int)target_len, .cap = 4096};
    /* The native execution consumes the materialized physical target. */
    controls(expected);
    ds41_even_receipt r; char err[256] = {0};
    int rc = ds41_even_policy(s, &full, archive_count, matches, 201, &r, err, sizeof(err));
    if (rc) fprintf(stderr, "actual padding boundary: %s\n", err);
    CHECK(!rc && r.published && r.execution.route == DS4_SCR_DENSE_REUSE);
    CHECK(r.logical_match.selected_ranges == matches && r.plan.range_count == matches);
    CHECK(r.execution.multi.committed_spans == matches && r.execution.multi.committed_tokens == matches * 128u);
    CHECK(r.new_pads == 1u && r.carried_pads == 0u && r.imported_pads == 0u);
    CHECK(r.planner_discovery_known && r.planner_discovered_matches >= matches &&
        r.planner_longest_match >= 128u);
    CHECK(r.physical_tokens == target_len + 1u && s->checkpoint.len == (int)(target_len + 1u));
    CHECK(!memcmp(s->checkpoint.v, expected, (target_len + 1u) * sizeof(int)));
    CHECK(s->checkpoint.v[10u + 5u] == 201); /* historical source pad content */
    CHECK(s->scr_coordinates.count == 1u && s->scr_coordinates.pads[0] == 9u &&
        s->checkpoint.v[s->scr_coordinates.pads[0]] == 201);
    CHECK(s->scr_coordinates.physical_tokens == target_len + 1u);
    /* Capture old-source facts before rebase; the local archive pointer may be
     * a retired predecessor afterward and is not dereferenced here. */
    CHECK(source_generation != 0u && r.execution.source_generation == source_generation);
    CHECK(r.imported_pads == 0u && s->scr_coordinates.identity != source_map_identity);
    CHECK(!ds4_session_scr_archive_release(s));
    free(e->vocab.token); e->vocab.token = NULL; e->vocab.n_vocab = 0;
    printf("actual padding boundary target=1336 stride=132 shift=1: one fresh pad, 10 aligned spans, historical source pad as content, publication passed\n");
}


static void actual_variant_limit_test(ds4_session *s, int *source, int *v) {
#ifdef DS41_SCR_PLANNER_TEST_HOOKS
    ds41_scr_planner_test_variant_limit = 1u;
    actual_no_padding_test(s, source, v, 1312u, 130u, 10u, 2u, 1u,
        CACHE_REUSE_VARIANT_LIMIT);
    ds41_scr_planner_test_variant_limit = 4096u;
    printf("actual limited incumbent: variant limit retained a valid materialized/rebased plan\n");
#else
    (void)s; (void)source; (void)v;
    fprintf(stderr, "actual variant-limit test requires -DDS41_SCR_PLANNER_TEST_HOOKS\n");
    exit(1);
#endif
}

static void actual_failure_setup(ds4_session *s, int *source, int *v,
        uint32_t archive_count, ds41_scr_archive **out) {
    ds41_gpu_graph *g = &s->ds41_graph;
    ds4_session_invalidate(s);
    s->checkpoint.len = 0; s->checkpoint_valid = false;
    s->scr_planner_budget_ms = 5000u;
    s->scr_capture_batch_chunks = true;
    for (uint32_t i = 0; i < 4096u; ++i) source[i] = 1000 + (int)i;
    controls(source);
    CHECK(!ds4_session_scr_archive_prepare(s, 8u, archive_count));
    ds41_scr_archive *a = g->archive; CHECK(a);
    for (uint32_t from = 8u; from < 8u + archive_count; from += 128u) {
        capture(g, from, 128u);
        memcpy(s->checkpoint.v, source, (size_t)(from + 128u) * sizeof(int));
        s->checkpoint.len = (int)(from + 128u); g->pos = from + 128u;
        s->checkpoint_valid = g->valid = true;
        CHECK(!ds4_session_scr_archive_flush(s));
    }
    CHECK(!ds4_session_scr_archive_seal(s, source + 8u, archive_count));
    for (uint32_t i = 0; i < 1312u; ++i) v[i] = 90000 + (int)i;
    for (uint32_t i = 0; i < 10u; ++i)
        memcpy(v + 8u + i * 130u, a->tokens + i * 130u, 128u * sizeof(int));
    *out = a;
}

static void actual_failure_case(ds4_session *s, int *source, int *v, int mode) {
    ds4_engine *e = s->engine;
    e->vocab.n_vocab = DS4_N_VOCAB;
    e->vocab.token = calloc(DS4_N_VOCAB, sizeof(*e->vocab.token)); CHECK(e->vocab.token);
    for (uint32_t i = 0; i < DS4_N_VOCAB; ++i)
        e->vocab.token[i] = (ds4_str){.ptr = "\xc4\x8a", .len = 2};
    ds41_scr_archive *a = NULL;
    actual_failure_setup(s, source, v, 2048u, &a);
    ds4_tokens full = {.v = v, .len = 1312, .cap = 4096};
    controls(v);
    if (mode == 0) cancel_now = true;
    else if (mode == 1) fail_stage = 1;
    else if (mode == 2) fail_install = 1;
    else if (mode == 3) fail_drain = 1;
    else if (mode == 4) fail_release = true;
    else if (mode == 5) s->scr_planner_budget_ms = 0u;
    ds41_even_receipt r; char err[256] = {0};
    int rc = ds41_even_policy(s, &full, 2048u, 10u, 201, &r, err, sizeof(err));
    if (rc) fprintf(stderr, "actual failure mode %d: %s\n", mode, err);
    if (mode == 0) {
        /* The outer cancellation checkpoint refused before invoking the Rust
         * planner, so no planner timing/status was measured. */
        CHECK(!r.planner_status_known && !r.planner_timing_known);
        CHECK(rc == DS4_SESSION_SYNC_INTERRUPTED);
        CHECK(!calls && !installs && !heads && !r.published && !r.target_mutated);
        CHECK(!r.execution.multi.committed_spans && !r.execution.multi.committed_tokens);
        CHECK(!s->scr_coordinate_charge && !s->scr_coordinates.count);
    } else if (mode == 5) {
        CHECK(r.planner_status_known && r.planner_timing_known);
        CHECK(rc == 0);
        CHECK(r.planner_status == CACHE_REUSE_ERR_BUDGET);
        CHECK(!r.planner_discovery_known && !r.planner_search_known);
        CHECK(r.planner_elapsed_ns > 0u && r.planner_preparation_ns > 0u);
        /* A zero planner budget has no reuse incumbent. The adapter takes its
         * ordinary refresh fallback, which may legitimately publish a fresh
         * checkpoint; no reuse installation may be claimed. */
        CHECK(r.execution.route == DS4_SCR_DENSE_REFRESH);
        CHECK(r.execution.reason == DS4_SCR_DENSE_MISS);
        CHECK(r.published && r.target_mutated && r.execution.complete_n);
        CHECK(r.execution.refresh.complete_n);
        CHECK(r.execution.refresh.prefix_completed == 0u &&
              r.execution.refresh.source_completed == 1312u &&
              r.execution.refresh.remainder_completed == 0u);
        CHECK(r.logical_match.selected_ranges == 0u && r.plan.range_count == 0u);
        CHECK(r.execution.multi.committed_spans == 0u && r.execution.multi.committed_tokens == 0u);
        CHECK(installs == 0u && s->checkpoint.len == 1312 &&
              !memcmp(s->checkpoint.v, v, 1312u * sizeof(int)));
    } else {
        CHECK(r.planner_status_known && r.planner_timing_known);
        CHECK(rc != 0 && !r.published && !r.execution.complete_n);
        CHECK(!r.execution.multi.committed_spans && !r.execution.multi.committed_tokens);
        CHECK(!r.execution.refresh.complete_n);
        CHECK(!r.execution.multi.published);
        CHECK(!s->checkpoint_valid || s->checkpoint.len == 0);
        CHECK(!s->scr_coordinates.count && !s->scr_coordinate_charge);
    }
    printf("actual failure mode %d rc=%d cancelled=%d stage=%d install=%d drain=%d release=%d planner_status=%u calls=%u installs=%u\n",
        mode, rc, cancel_now, fail_stage, fail_install, fail_drain, fail_release,
        r.planner_status, calls, installs);
    free(e->vocab.token); e->vocab.token = NULL; e->vocab.n_vocab = 0;
    fail_stage = fail_install = 0; fail_drain = fail_release = cancel_now = false;
    s->scr_planner_budget_ms = 5000u;
    CHECK(!ds4_session_scr_archive_release(s));
    ds4_session_invalidate(s);
}

static void actual_failure_test(ds4_session *s, int *source, int *v) {
    actual_failure_case(s, source, v, 0);
    actual_failure_case(s, source, v, 1);
    actual_failure_case(s, source, v, 2);
    actual_failure_case(s, source, v, 3);
    actual_failure_case(s, source, v, 4);
    actual_failure_case(s, source, v, 5);
    printf("actual adapter cancellation/stage/install/rebase-failure and zero-budget timing checks passed\n");
}

int main(void) {
    CHECK(ds4_gpu_init()); g_ds4_shape = DS4_SHAPE_FLASH41;
    ds4_engine e = {.backend=DS4_BACKEND_METAL};
    ds4_session s = {.engine=&e,.ctx_size=4096,.ds41_graph_ready=true,
        .progress=progress,.display_progress=progress,.cancel=cancel,
        .scr_planner_budget_ms=5000u}; session = &s;
    s.progress_ud = &s; s.display_progress_ud = &callbacks;
    ds41_gpu_graph *g = &s.ds41_graph; g->ctx = 4096; g->tp_world = 1; g->prefill_cap = 128; g->valid = true;
    g->table[0].fd = g->table[1].fd = -1;
    g->token_map = malloc(DS4_N_VOCAB*sizeof(uint32_t)); s.logits = malloc(DS4_N_VOCAB*sizeof(float));
    s.checkpoint.v = malloc(4096*sizeof(int)); s.checkpoint.cap = 4096;
    CHECK(g->token_map && s.logits && s.checkpoint.v);
    for (uint32_t i = 0; i < DS4_N_VOCAB; ++i) g->token_map[i] = i ^ 73u;
    uint64_t live = 0;
    for (uint32_t o = 0; o < 4; ++o) {
        uint32_t ratio = o < 3 ? 2u : 1u;
        g->compressed[o] = ds4_gpu_tensor_alloc(4096ull/ratio*512u*4u);
        g->index_cache[o] = ds4_gpu_tensor_alloc(4096ull/ratio*128u*4u);
        CHECK(g->compressed[o] && g->index_cache[o]); live += 4096ull/ratio*640u*4u;
    }
    for (uint32_t l = 0; l < 40; ++l) {
        g->window[l] = ds4_gpu_tensor_alloc(sizeof(values)); CHECK(g->window[l]); live += sizeof(values);
    }
    g->allocation_bytes = e.ds41_session_bytes = live;
    rows = ds4_gpu_tensor_alloc(sizeof(values)); indices = ds4_gpu_tensor_alloc(DS41_RAW_ARENA_TOKENS*128u*4u);
    CHECK(rows && indices);
    static int source[4096], v[4096];
    for (uint32_t i = 0; i < 4096; ++i) source[i] = v[i] = (int)i+1000;
    actual_no_padding_test(&s, source, v, 1312u, 130u, 10u, 10u, 10u, CACHE_REUSE_COMPLETE);
    actual_no_padding_test(&s, source, v, 1312u, 130u, 6u, 6u, 6u, CACHE_REUSE_COMPLETE);
    actual_no_padding_test(&s, source, v, 1325u, 132u, 10u, 10u, 9u, CACHE_REUSE_COMPLETE);
    actual_variant_limit_test(&s, source, v);
    actual_failure_test(&s, source, v);
    actual_padding_boundary_test(&s, source, v);
    ds4_gpu_tensor_free(rows); ds4_gpu_tensor_free(indices);
    for (uint32_t o=0;o<4;++o) { ds4_gpu_tensor_free(g->compressed[o]); ds4_gpu_tensor_free(g->index_cache[o]); }
    for (uint32_t l=0;l<40;++l) ds4_gpu_tensor_free(g->window[l]);
    free(g->token_map); free(s.logits); free(s.checkpoint.v); session = NULL;
    puts("actual cache-reuse-utils planner: K bounds, padding boundary, variant limit, failure modes passed (host storage, simulated rows)");
    return 0;
}
